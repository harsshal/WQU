
def palindrome(str):
    n = len(str)

    # table where we will calculate palindrome substrings in right top corner
    table = [[0 for column in range(n)] for row in range(n)]

    for i in range(n):
        table[i][i] = str[i]

    for sub_len in range(2, n+1):
        for row in range(n-sub_len+1):
            # we need to populate table along diagonal first
            # and then work our way to right top corner
            column = row+sub_len-1
            if str[row] == str[column] and sub_len==2 :
                table[row][column] = str[row]+str[column]
            elif str[row] == str[column]:
                table[row][column] = str[row] + table[row+1][column-1] + str[column];
            else:
                if(len(table[row][column-1]) > len(table[row+1][column]) ):
                    table[row][column] = table[row][column-1]
                else:
                    table[row][column] = table[row+1][column]
    return table[0][n-1]

print(palindrome('aibohphobia'))
