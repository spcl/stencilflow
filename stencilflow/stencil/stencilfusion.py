import ast
import sys
import dace
from dace.transformation.pattern_matching import Transformation
from dace.transformation.dataflow import MapFission
from typing import Any, Dict, Set
import warnings

from dace import registry, sdfg as sd, symbolic
from dace.properties import make_properties
from dace.graph import nodes, nxutil, labeling
from stencilflow.stencil.stencil import Stencil


class ReplaceSubscript(ast.NodeTransformer):
    def __init__(self, subscript: str, new_name: str):
        self.src = subscript
        self.dst = new_name

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id == self.src:
            return ast.copy_location(ast.Name(id=self.dst), node)
        return self.generic_visit(node)


@registry.autoregister_params(singlestate=True)
@make_properties
class StencilFusion(Transformation):
    """ Transformation that nests a one-dimensional map into a stencil,
        including it in the computational domain. """

    _stencil_a = Stencil('')
    _stencil_b = Stencil('')
    _tmp_array = nodes.AccessNode('_')

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(StencilFusion._stencil_a,
                                   StencilFusion._tmp_array,
                                   StencilFusion._stencil_b)
        ]

    @staticmethod
    def match_to_str(graph, candidate):
        stencil_a: Stencil = graph.node(candidate[StencilFusion._stencil_a])
        stencil_b: Stencil = graph.node(candidate[StencilFusion._stencil_b])
        return '%s -> %s' % (stencil_a.label, stencil_b.label)

    @staticmethod
    def can_be_applied(graph: dace.SDFGState,
                       candidate: Dict[Any, int],
                       expr_index: int,
                       sdfg: dace.SDFG,
                       strict=False):
        stencil_a: Stencil = graph.node(candidate[StencilFusion._stencil_a])
        stencil_b: Stencil = graph.node(candidate[StencilFusion._stencil_b])
        array: nodes.AccessNode = graph.node(
            candidate[StencilFusion._tmp_array])

        # Ensure the stencil shapes match
        if len(stencil_a.shape) != len(stencil_b.shape):
            return False
        if any(sa != sb for sa, sb in zip(stencil_a.shape, stencil_b.shape)):
            return False

        # Ensure that the transient is not used anywhere else and can be
        # removed
        if len(graph.all_edges(array)) != 2:
            return False
        if not sdfg.arrays[array.data].transient:
            return False
        if (len([
                n for state in sdfg.nodes() for n in state.nodes()
                if isinstance(n, nodes.AccessNode) and n.data == array.data
        ]) > 1):
            return False

        # Ensure that second stencil only has one input access of the
        # candidate transient to remove
        edge = graph.out_edges(array)[0]
        if len(stencil_b.accesses[edge.dst_conn][1]) > 1:
            return False

        # TODO: Remove check once stencils can be offset
        if any(a != 0 for a in stencil_b.accesses[edge.dst_conn][1][0]):
            return False

        # Code languages must match
        if stencil_a._code['language'] != stencil_b._code['language']:
            return False

        # TODO: Boundary condition matching checks

        return True

    def apply(self, sdfg: dace.SDFG):
        graph: dace.SDFGState = sdfg.node(self.state_id)
        stencil_a: Stencil = graph.node(
            self.subgraph[StencilFusion._stencil_a])
        stencil_b: Stencil = graph.node(
            self.subgraph[StencilFusion._stencil_b])
        array: nodes.AccessNode = graph.node(
            self.subgraph[StencilFusion._tmp_array])

        intermediate_name = graph.in_edges(array)[0].src_conn
        intermediate_name_b = graph.out_edges(array)[0].dst_conn

        # Replace outputs of first stencil with outputs of second stencil
        # In node and in connectors, reconnect
        stencil_a.output_fields = stencil_b.output_fields
        stencil_a.boundary_conditions = stencil_b.boundary_conditions
        for edge in list(graph.out_edges(stencil_a)):
            if edge.src_conn == intermediate_name:
                graph.remove_edge(edge)
                stencil_a._out_connectors.remove(intermediate_name)
        for edge in graph.out_edges(stencil_b):
            stencil_a.add_out_connector(edge.src_conn)
            graph.add_edge(stencil_a, edge.src_conn, edge.dst, edge.dst_conn,
                           edge.data)

        # Add other stencil inputs of the second stencil to the first
        # In node and in connectors, reconnect
        for edge in graph.in_edges(stencil_b):
            # Skip edge to remove
            if edge.dst_conn == intermediate_name_b:
                continue
            if edge.dst_conn not in stencil_a.accesses:
                stencil_a.accesses[edge.dst_conn] = stencil_b.accesses[
                    edge.dst_conn]
                stencil_a.add_in_connector(edge.dst_conn)
                graph.add_edge(edge.src, edge.src_conn, stencil_a,
                               edge.dst_conn, edge.data)
            else:
                # If same input is accessed in both stencils, only append the
                # inputs that are new to stencil_a
                for access in stencil_b.accesses[edge.dst_conn][1]:
                    if access not in stencil_a.accesses[edge.dst_conn][1]:
                        stencil_a.accesses[edge.dst_conn][1].append(access)

        # Add second stencil's statements to first stencil, replacing the input
        # to the second stencil with the name of the output access
        if stencil_a._code['language'] == dace.Language.Python:
            # Replace first stencil's output with connector name
            for i, stmt in enumerate(stencil_a.code):
                stencil_a._code['code_or_block'][i] = ReplaceSubscript(
                    intermediate_name, intermediate_name_b).visit(stmt)

            # Append second stencil's contents, using connector name instead of
            # accessing the intermediate transient
            # TODO: Use offsetted stencil
            for i, stmt in enumerate(stencil_b.code):
                stencil_a._code['code_or_block'].append(
                    ReplaceSubscript(intermediate_name_b,
                                     intermediate_name_b).visit(stmt))

        elif stencil_a._code['language'] == dace.Language.CPP:
            raise NotImplementedError
        else:
            raise ValueError('Unrecognized language: %s' %
                             stencil_a._code['language'])

        # Remove array from graph
        graph.remove_node(array)
        del sdfg.arrays[array.data]

        # Remove 2nd stencil
        graph.remove_node(stencil_b)


if __name__ == '__main__':
    from stencilflow.sdfg_to_stencilflow import (standardize_data_layout,
                                                 remove_extra_subgraphs)
    from stencilflow.stencil.nestk import NestK
    from dace.transformation.interstate import StateFusion

    sdfg: dace.SDFG = dace.SDFG.from_file(sys.argv[1])

    sdfg.apply_transformations_repeated([MapFission])

    # Partial canonicalization
    remove_extra_subgraphs(sdfg)
    standardize_data_layout(sdfg)

    sdfg.apply_transformations_repeated([NestK])
    sdfg.apply_transformations_repeated([StateFusion])
    sdfg.apply_strict_transformations()

    # After graph is preprocessed, run StencilFusion
    sdfg.apply_transformations_repeated([StencilFusion])
    sdfg.save('fused.sdfg')
