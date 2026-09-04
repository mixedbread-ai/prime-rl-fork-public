from importlib import import_module
from types import ModuleType
from typing import Any, Callable, NamedTuple


class VllmServerAPI(NamedTuple):
    entry: ModuleType
    init_app_state: Callable[..., Any]
    build_app: Callable[..., Any]
    make_arg_parser: Callable[..., Any]
    validate_parsed_serve_args: Callable[..., Any]


def load_vllm_server_api() -> VllmServerAPI:
    try:
        cli_args = import_module("vllm.entrypoints.launchers.cli_args")
    except ModuleNotFoundError as error:
        if error.name not in {"vllm.entrypoints.launchers", "vllm.entrypoints.launchers.cli_args"}:
            raise
        entry = import_module("vllm.entrypoints.openai.api_server")
        cli_args = import_module("vllm.entrypoints.openai.cli_args")
        init_app_state = entry.init_app_state
        build_app = entry.build_app
    else:
        entry = import_module("vllm.entrypoints.launchers.api_server.entry")
        app_state = import_module("vllm.entrypoints.launchers.api_server.app_state")
        app = import_module("vllm.entrypoints.launchers.app")
        init_app_state = app_state.init_app_state
        build_app = app.build_app

    return VllmServerAPI(
        entry=entry,
        init_app_state=init_app_state,
        build_app=build_app,
        make_arg_parser=cli_args.make_arg_parser,
        validate_parsed_serve_args=cli_args.validate_parsed_serve_args,
    )


def install_server_overrides(
    api: VllmServerAPI,
    init_app_state: Callable[..., Any],
    build_app: Callable[..., Any],
) -> None:
    api.entry.init_app_state = init_app_state
    api.entry.build_app = build_app
