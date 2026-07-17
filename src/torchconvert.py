# torch/csrc positional PyTypeObject initializers -> designated, multi-file.
# Generalizes wasthon4/src/dtconvert.py (the datetime pass): designated is
# equivalent to positional C99, and em++ accepts out-of-order + mixed
# positional/designated in C++ (warnings -Wreorder-init-list/-Wc99-designator
# only). Extensions over dtconvert needed for torch's format:
#   - slot expressions spanning several lines (tp_flags OR-chains),
#   - interstitial // comments (NOLINT) preserved,
#   - non-static `PyTypeObject X = {` definitions,
#   - fields absent from wasthon's struct _typeobject (tp_reserved): dropped
#     when the initializer is null, kept + WARNED otherwise.
# The PyVarObject_HEAD_INIT(...) head is left as-is (wasthon's macro expands
# to `{0}, (size),` covering ob_refcnt/tp_free positionally); a HEAD_INIT
# whose first argument is `&SomeMeta` is REPORTED: wasthon's macro discards
# it, so module init must call Py_SET_TYPE(&Type, &SomeMeta).
#
# Usage: python3 torchconvert.py <wasthon.h> [--dry-run] <file.cpp>...
import re
import sys

args = sys.argv[1:]
dry = '--dry-run' in args
if dry:
    args.remove('--dry-run')
wasthon_h, files = args[0], args[1:]

# Field order is irrelevant (designated), we only need the set of names.
hsrc = open(wasthon_h).read()
body = hsrc[hsrc.index('struct _typeobject {'):]
body = body[:body.index('\n};')]
known = set(re.findall(r'\b(tp_\w+|ob_refcnt)\b', body))
assert 'tp_name' in known and 'tp_flags' in known and 'tp_getset' in known

NULLS = {'0', 'NULL', 'nullptr'}
marker = re.compile(r'/\*\s*(tp_\w+)\s*\*/')
opener = re.compile(r'(?:static\s+)?PyTypeObject\s+(\w+)\s*=\s*\{')
head_meta = re.compile(r'PyVarObject_HEAD_INIT\(\s*&(\w+)')

total_slots = total_types = 0
warnings = []

def convert_block(fname, tname, body):
    """body = text between the opening '{' and the '\n};'. Returns new body."""
    global total_slots
    marks = [mk for mk in marker.finditer(body)
             if '//' not in body[body.rfind('\n', 0, mk.start()) + 1:mk.start()]]
    if not marks:
        return body  # already designated (or empty): untouched
    out = [body[:marks[0].start()]]  # head: HEAD_INIT line(s), kept verbatim
    m = head_meta.search(out[0])
    if m:
        warnings.append('%s: %s has metatype &%s -> needs Py_SET_TYPE at init'
                        % (fname, tname, m.group(1)))
    prev_end = None
    for mk in marks:
        field = mk.group(1)
        if prev_end is None:
            seg = out.pop(0)
        else:
            seg = body[prev_end:mk.start()]
        prev_end = mk.end()
        # split leading blank/comment/HEAD_INIT lines from the expression
        # (the first slot's segment starts with the PyVarObject_HEAD_INIT
        # line, which must stay verbatim ahead of the designated list)
        lines = seg.split('\n')
        lead = []
        while lines and (not lines[0].strip()
                         or lines[0].lstrip().startswith('//')
                         or lines[0].lstrip().startswith('#')
                         or 'HEAD_INIT' in lines[0]):
            lead.append(lines.pop(0))
        expr = '\n'.join(lines).rstrip().rstrip(',').strip()
        # detach trailing block comments (torch: `nullptr,\n /* will be
        # assigned in init */ /* tp_methods */`) — re-emitted after the comma
        trail = ''
        m3 = re.search(r'(?:\s*/\*(?:[^*]|\*(?!/))*\*/)+$', expr)
        if m3:
            trail = ' ' + expr[m3.start():].strip()
            expr = expr[:m3.start()].rstrip().rstrip(',').strip()
        assert expr, '%s: %s: empty expr for %s' % (fname, tname, field)
        assert '\n#' not in expr, ('%s: %s: preprocessor directive inside '
                                   '%s expression' % (fname, tname, field))
        indent = re.match(r'\s*', lines[0]).group(0)
        if field not in known and expr in NULLS:
            # no such field in wasthon, null anyway: drop the line, keep
            # any real comments riding above it
            if any(ln.strip() for ln in lead):
                out.append('\n'.join(lead))
            continue
        if field not in known:
            warnings.append('%s: %s: UNKNOWN field %s = %s (kept; append it '
                            'to wasthon.h struct _typeobject)'
                            % (fname, tname, field, expr))
        out.append('\n'.join(lead) + '\n' if lead else '')
        out.append('%s.%s = %s,%s' % (indent, field, expr, trail))
        total_slots += 1
    out.append(body[prev_end:])  # tail after last marker, verbatim
    return ''.join(out)

for path in files:
    src = open(path).read()
    out, pos, nconv = [], 0, 0
    while True:
        m = opener.search(src, pos)
        if not m:
            out.append(src[pos:])
            break
        end = re.compile(r'\n[ \t]*\};').search(src, m.end()).start()
        newbody = convert_block(path, m.group(1), src[m.end():end])
        if newbody != src[m.end():end]:
            nconv += 1
        out.append(src[pos:m.end()])
        out.append(newbody)
        pos = end
    if nconv:
        total_types += nconv
        if not dry:
            open(path, 'w').write(''.join(out))
    print('%-60s types converted: %d' % (path, nconv))

print('\nTOTAL: %d types, %d slots%s' % (total_types, total_slots,
                                         ' (DRY RUN)' if dry else ''))
for w in warnings:
    print('WARN:', w)
