def change_number(number: int):
    relative = 195883
    return number - relative


with open('bkp/output.ann', 'r') as f:
    lines = f.readlines()
final_lines = []

for line in lines:
    line = line.replace('\t', ' ').replace('\n', ' ')
    items = line.split(' ')
    if 'T' in items[0]:
        items_clean = []
        for item in items:
            if item != '':
                items_clean.append(item)
        items = items_clean
        number_items = (len(items) - 5) // 2 + 1
        items[2] = str(change_number(int(items[2])))
        for i in range(3, 3 + number_items - 1):
            item_1 = items[i].split(';')[0]
            item_2 = items[i].split(';')[1]
            items[i] = f'{change_number(int(item_1))};{change_number(int(item_2))}'
        items[3 + number_items - 1] = str(change_number(int(items[3 + number_items - 1])))
        final_line = items[0] + '\t' + items[1]
        for i in range(2, len(items) - number_items):
            final_line += ' ' + items[i]
        final_line += '\t' + items[len(items) - number_items]
        for i in range(len(items) - number_items + 1, len(items)):
            final_line += ' ' + items[i]
        final_lines.append(final_line + '\n')
    else:
        final_lines.append(line + '\n')

with open('output.ann', 'w') as f:
    f.writelines(final_lines)
