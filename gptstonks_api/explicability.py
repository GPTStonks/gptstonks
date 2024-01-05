def add_context_to_output(output: str, tools_executed: list[str]) -> str:
    openbb_context = "> Context retrieved using OpenBB."
    complete_context = (
        f"> Context retrieved using {','.join(tools_executed)}."
        if len(tools_executed) > 0
        else "> Answer generated with the AI's own knowledge."
    )

    return (
        output.replace(openbb_context, complete_context)
        if openbb_context in output
        else f"{complete_context}\n\n{output}"
    )
