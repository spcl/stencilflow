import sys
import dace
from dace.transformation.pattern_matching import Transformation
from dace.transformation.dataflow import MapFission, MapCollapse
from typing import Any, Dict
import warnings

from dace import registry, sdfg as sd
from dace.properties import make_properties
from dace.graph import nodes, nxutil, labeling
from stencilflow.stencil.stencil import Stencil


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
    def can_be_applied(graph: dace.SDFGState, candidate: Dict[Any, int],
                       expr_index: int, sdfg: dace.SDFG, strict=False):
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
                                warnings.warn('k expression is not trivial')
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
                graph.add_edge(stencil, child.edge.src_conn,
                               edge.dst, edge.dst_conn, memlet)

        # Remove map
        graph.remove_nodes_from([map_entry, map_exit])

        # Reshape stencil node computation based on nested map range
        stencil.shape[dim_index] = map_entry.map.range.num_elements()


if __name__ == '__main__':
    sdfg = dace.SDFG.from_file(sys.argv[1])

    sdfg.apply_transformations_repeated([MapFission, NestK])
    dace.propagate_labels_sdfg(sdfg)
    sdfg.apply_strict_transformations()
    sdfg.save('nested.sdfg')

    #Stencil.default_implementation = 'CPU'
    #sdfg.expand_library_nodes()
