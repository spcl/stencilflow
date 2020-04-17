from dace import registry, InterstateEdge
from dace.properties import make_properties
from dace.transformation.interstate.loop_detection import DetectLoop

@registry.autoregister
@make_properties
class RemoveLoop(DetectLoop):
    """ Unrolls a state machine for-loop into multiple states """

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        if not DetectLoop.can_be_applied(graph, candidate, expr_index, sdfg,
                                         strict):
            return False
        return True

    def apply(self, sdfg):

        # Obtain loop information
        guard: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_guard])
        begin: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_begin])
        after_state: sd.SDFGState = sdfg.node(
            self.subgraph[DetectLoop._exit_state])
        # Remove edge from guard to after state
        for e in sdfg.out_edges(guard):
            if e.dst == after_state:
                sdfg.remove_edge(e)
        # Find the backedge and move it
        edges = list(sdfg.out_edges(begin))
        while len(edges) > 0:
            e = edges.pop()
            if e.dst == guard:
                # Bypass guard and go directly to after-state
                sdfg.remove_edge(e)
                sdfg.add_edge(e.src, after_state, InterstateEdge())
            else:
                edges += list(sdfg.out_edges(e.dst))
