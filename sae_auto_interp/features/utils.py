def vis(
    record,
    tokenizer,
    n=10,
    threshold=0.0,
) -> str:

    from IPython.core.display import display, HTML

    def _to_string(tokens, activations):
        result = []
        i = 0

        max_act = max(activations)
        _threshold = max_act * threshold

        while i < len(tokens):
            if activations[i] > _threshold:
                result.append("<mark>")
                while i < len(tokens) and activations[i] > _threshold:
                    result.append(tokens[i])
                    i += 1
                result.append("</mark>")
            else:
                result.append(tokens[i])
                i += 1
        return "".join(result)
    
    strings = []

    for example in record.examples[:n]:

        str_toks = tokenizer.batch_decode(
            example.tokens
        )

        strings.append(
            _to_string(
                str_toks, 
                example.activations
            )
        )

    display(HTML("<br><br>".join(strings)))