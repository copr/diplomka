def load(filename, lines_to_skip):
    data = []
    line_count = 0
    with open(filename) as f:
        for line in f:
            if line_count >= lines_to_skip or len(line) == 0:
                parsed_line = tuple(filter(lambda x: x != '', line[:-1].split(' ')))
                data.append(parsed_line)
            line_count += 1
    return data
