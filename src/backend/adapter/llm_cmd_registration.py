import inspect

llm_commands = {}


def llm_command(func):
    llm_commands[func.__name__] = func
    return func


def annotation_to_string(annotation) -> str:
    if annotation is None:
        return "None"

    # generic?
    if not hasattr(annotation, "__origin__"):
        return getattr(annotation, "__name__", annotation)

    origin = annotation.__origin__
    args = annotation.__args__
    type_str = f"{origin.__name__}[{', '.join(arg.__name__ for arg in args)}]"
    return type_str


def get_llm_commands():
    res = []
    for cmnd_name, llm_command in llm_commands.items():
        params = []
        sig = inspect.signature(llm_command)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            params += [f"{name}: {param.annotation.__name__}"]

        returnType = "None" if sig.return_annotation is None else sig.return_annotation

        signature = (
            f"{cmnd_name}({', '.join(params)}) -> {annotation_to_string(returnType)}"
        )
        signature = signature.replace("_empty", "<unspecified>")
        docs = llm_command.__doc__.strip("\n")
        res += [f"{signature}\n{docs}"]

    s = "\n\n".join(res)
    s = s.replace("__main__.", "")  # remove unnecessary main references
    return s
