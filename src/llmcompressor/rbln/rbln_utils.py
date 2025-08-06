from dataclasses import dataclass
import os
from typing import Any, Callable

import torch
from torch.fx import GraphModule
from torch.fx.graph import PythonCode
import rebel


ENFORCE_EAGER = os.environ.get("RBLN_COMPILE", "0") == "0"


@dataclass
class RBLNSubgraph:
    """
    Dataclass specifying an executable subgraph of a model graph

    :param graph: subgraph of model graph
    :param input_names: argument names of the compiled forward function
    :param consumed_names: argument names which are not used by any subsequent subgraphs
        and can therefore be deleted from the intermediates cache
    """

    graph_module: GraphModule
    input_names: set[str]
    consumed_names: set[str]
    _code: PythonCode | None = None
    _compiled_module: Callable | None = None # type: ignore

    def __post_init__(self):
        if not ENFORCE_EAGER:
            raise NotImplementedError("RBLN compilation is not implemented yet.")
            # self._compiled_module = torch.compile(self.graph_module, backend="rbln", dynamic=False)
            # TODO: clear compilation cache after execution of each subgraph to save device memory.(torch._dynamo.reset())

    def forward(self, *args, **kwargs) -> dict[str, Any]:
        """
        Execute the operations within the subgraph

        :param \\*args: argument inputs to subgraph forward function
        :param \\**kwargs: keyword inputs to subgraph forward function
        :return keyword outputs of subgraph forward function (non-consumed variables):
        """
        if self._code is None:
            self._code = self.graph_module.graph.python_code("self")
            exec(self._code.src, self._code.globals)

        forward_fn = self._code.globals.get("forward")

        try:
            if ENFORCE_EAGER:
                outputs = forward_fn(*args, **kwargs)
            else:
                assert self._compiled_module is not None
                outputs = self._compiled_module(*args, **kwargs)
        except Exception as exception:
            from llmcompressor.pipelines.sequential.helpers import add_line_numbers
            raise RuntimeError(
                "Raised an exception during execution of the following code:\n"
                f"```\n{add_line_numbers(self._code.src)}\n```\n"
                "This is likely due to a violation of shape assumptions made when "
                "tracing"
            ) from exception

        return outputs
