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


class DimensionAdder(ast.NodeTransformer):
    """ Adds a dimension in a Python AST to all subscripts of the specified
        arrays. """
    def __init__(self, names: Set[str], dim_index: int, value: int = 0):
        self.names = names
        self.dim = dim_index
        self.value = value

    def visit_Subscript(self, node: ast.Subscript):
        if not isinstance(node.value, ast.Name):
            raise TypeError('Only subscripts of variables are supported')

        varname = node.value.id

        # Add dimension to correct location
        if varname in self.names:
            node.slice.value.elts.insert(
                self.dim,
                ast.copy_location(
                    ast.parse(str(self.value)).body[0].value,
                    node.slice.value.elts[0]))
            return node

        return self.generic_visit(node)


@registry.autoregister_params(singlestate=True)
@make_properties
class NestK(Transformation):
    """ Transformation that nests a one-dimensional map into a stencil,
        including it in the computational domain. """

    _map_entry = nodes.MapEntry(nodes.Map('', [], []))
    _stencil = Stencil('')

    @staticmethod
    def expressions():
        return [nxutil.node_path_graph(NestK._map_entry, NestK._stencil)]

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry: nodes.MapEntry = graph.node(candidate[NestK._map_entry])
        stencil: Stencil = graph.node(candidate[NestK._stencil])
        return '%s into %s' % (map_entry.map.label, stencil.label)

    @staticmethod
    def can_be_applied(graph: dace.SDFGState,
                       candidate: Dict[Any, int],
                       expr_index: int,
                       sdfg: dace.SDFG,
                       strict=False):
        map_entry: nodes.MapEntry = graph.node(candidate[NestK._map_entry])
        stencil: Stencil = graph.node(candidate[NestK._stencil])

        if len(map_entry.map.params) != 1:
            return False
        if sd.has_dynamic_map_inputs(graph, map_entry):
            return False
        pname = map_entry.map.params[0]  # Usually "k"
        dim_index = None

        for edge in graph.out_edges(map_entry):
            if edge.dst != stencil:
                return False

        for edge in graph.all_edges(stencil):
            if edge.data.data is None:  # Empty memlet
                continue
            # TODO: Use bitmap to verify lower-dimensional arrays
            if len(edge.data.subset) == 3:
                for i, rng in enumerate(edge.data.subset.ndrange()):
                    for r in rng:
                        if (len(r.free_symbols) == 1
                                and pname in map(str, r.free_symbols)):
                            if dim_index is not None and dim_index != i:
                                # k dimension must match in all memlets
                                return False
                            if str(r) != pname:
                                if symbolic.issymbolic(r -
                                                       symbolic.symbol(pname)):
                                    warnings.warn('k expression is nontrivial')
                            dim_index = i

        # No nesting dimension found
        if dim_index is None:
            return False

        # Ensure the stencil shape is 1 for the found dimension
        if stencil.shape[dim_index] != 1:
            return False

        return True

    def apply(self, sdfg: dace.SDFG):
        graph: dace.SDFGState = sdfg.node(self.state_id)
        map_entry: nodes.MapEntry = graph.node(self.subgraph[NestK._map_entry])
        stencil: Stencil = graph.node(self.subgraph[NestK._stencil])

        # Find dimension index and name
        pname = map_entry.map.params[0]
        dim_index = None
        for edge in graph.all_edges(stencil):
            if edge.data.data is None:  # Empty memlet
                continue

            if len(edge.data.subset) == 3:
                for i, rng in enumerate(edge.data.subset.ndrange()):
                    for r in rng:
                        if (len(r.free_symbols) == 1
                                and pname in map(str, r.free_symbols)):
                            dim_index = i
                            break
                    if dim_index is not None:
                        break
                if dim_index is not None:
                    break
        ###

        map_exit = graph.exit_nodes(map_entry)[0]

        # Reconnect external edges directly to stencil node
        for edge in graph.in_edges(map_entry):
            # Find matching internal edges
            tree = graph.memlet_tree(edge)
            for child in tree.children:
                memlet = labeling.propagate_memlet(graph, child.edge.data,
                                                   map_entry, False)
                graph.add_edge(edge.src, edge.src_conn, stencil,
                               child.edge.dst_conn, memlet)
        for edge in graph.out_edges(map_exit):
            # Find matching internal edges
            tree = graph.memlet_tree(edge)
            for child in tree.children:
                memlet = labeling.propagate_memlet(graph, child.edge.data,
                                                   map_entry, False)
                graph.add_edge(stencil, child.edge.src_conn, edge.dst,
                               edge.dst_conn, memlet)

        # Remove map
        graph.remove_nodes_from([map_entry, map_exit])

        # Reshape stencil node computation based on nested map range
        stencil.shape[dim_index] = map_entry.map.range.num_elements()

        # Add dimensions to access and output fields
        add_dims = set()
        for edge in graph.in_edges(stencil):
            if edge.data.data and len(edge.data.subset) == 3:
                if stencil.accesses[edge.dst_conn][0][dim_index] is False:
                    add_dims.add(edge.dst_conn)
                stencil.accesses[edge.dst_conn][0][dim_index] = True
        for edge in graph.out_edges(stencil):
            if edge.data.data and len(edge.data.subset) == 3:
                if stencil.output_fields[edge.src_conn][0][dim_index] is False:
                    add_dims.add(edge.src_conn)
                stencil.output_fields[edge.src_conn][0][dim_index] = True
        # Change all instances in the code as well
        if stencil._code['language'] != dace.Language.Python:
            raise ValueError(
                'For NestK to work, Stencil code language must be Python')
        stencil.code = DimensionAdder(add_dims,
                                      dim_index).visit(stencil.code[0])
        stencil.code.as_string = None  # Force regeneration


if __name__ == '__main__':
    from stencilflow.sdfg_to_stencilflow import standardize_data_layout

    sdfg: dace.SDFG = dace.SDFG.from_file(sys.argv[1])

    sdfg.apply_transformations_repeated([MapFission])
    standardize_data_layout(sdfg)
    sdfg.apply_transformations_repeated([NestK])
    dace.propagate_labels_sdfg(sdfg)
    sdfg.apply_strict_transformations()
    sdfg.save('nested.sdfg')
    # Stencil.default_implementation = 'CPU'
    # sdfg.expand_library_nodes()
