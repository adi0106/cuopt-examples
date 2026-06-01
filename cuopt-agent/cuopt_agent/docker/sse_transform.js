// SSE body filter: converts NAT's `intermediate_data:` events into
// chat-completion-shaped `data:` events so a single UI that only understands
// `data:` chunks can render reasoning, planning, and tool execution as
// streaming markdown alongside the final answer.
//
// Each NAT intermediate event looks like:
//   intermediate_data: {"id","parent_id","type","name","payload", ...}
//
// We emit a chat.completion.chunk with delta.content set to a markdown header
// (`**[<name>]**`) followed by the payload, so the downstream stream is one
// continuous markdown document.
//
// Stateful across invocations: nginx calls this filter once per upstream
// chunk, and SSE events may span chunk boundaries. We buffer any trailing
// partial line in $sse_buf (declared with `js_var` in nginx.conf) until the
// next call.

const INTERMEDIATE_PREFIX = 'intermediate_data: ';

function reshapeIntermediate(line) {
    const jsonStr = line.slice(INTERMEDIATE_PREFIX.length);
    let step;
    try {
        step = JSON.parse(jsonStr);
    } catch (e) {
        // Malformed JSON — emit the original line so nothing is silently lost.
        return line + '\n';
    }

    const name = step.name || step.type || 'step';
    const payload = step.payload != null ? String(step.payload) : '';
    const content = '\n\n**[' + name + ']**\n' + payload + '\n';

    const chunk = {
        id: step.id || '',
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model: 'unknown-model',
        choices: [{
            index: 0,
            delta: { content: content, role: 'assistant' },
            finish_reason: null,
        }],
    };

    return 'data: ' + JSON.stringify(chunk) + '\n';
}

function transform(r, data, flags) {
    const combined = (r.variables.sse_buf || '') + data;

    // Process only complete lines; buffer any trailing partial for next call.
    const lastNewline = combined.lastIndexOf('\n');
    let processable, leftover;
    if (lastNewline === -1) {
        processable = '';
        leftover = combined;
    } else {
        processable = combined.slice(0, lastNewline + 1);
        leftover = combined.slice(lastNewline + 1);
    }

    let out = '';
    if (processable) {
        const lines = processable.split('\n');
        // `"a\nb\n".split('\n')` -> `['a', 'b', '']`; the trailing '' belongs
        // to the final newline already accounted for, so skip it.
        for (let i = 0; i < lines.length - 1; i++) {
            const line = lines[i];
            if (line.startsWith(INTERMEDIATE_PREFIX)) {
                out += reshapeIntermediate(line);
            } else {
                out += line + '\n';
            }
        }
    }

    if (flags.last) {
        // Final flush: drain any leftover partial line untouched.
        if (leftover) out += leftover;
        r.variables.sse_buf = '';
    } else {
        r.variables.sse_buf = leftover;
    }

    r.sendBuffer(out, flags);
}

export default { transform };
