"""
QUESTION#1
Given two strings s and t, determine whether some anagram of t is a substring of s.
For example: if s = "udacity" and t = "ad", then the function returns True. Your function
definition should look like: question1(s, t) and return a boolean True or False.
"""

def char_count(a):
    '''helper function to calculate store elements count in dict'''
    char_count_dict = {}
    for i in a:
        if i not in char_count_dict:
            char_count_dict[i] = 1
        else:
            char_count_dict[i] += 1

    return char_count_dict

def question1(s, t):
    char_count_s = char_count(s)
    char_count_t = char_count(t)
    count = 0
    for k, v in char_count_t.items():
        if k not in char_count_s:
            continue
        elif v == char_count_s[k]:
            count += 1
        else:
            continue
    if count == len(char_count_t):
        return True
    else:
        return False

def main():
    #Testcase1 - Expected True because Anagram exists
    s = "udacity"
    t = "ad"
    print(question1(s, t))
    # Testcase2 - Expected True because Anagram exists
    s = "udacity"
    t = "acid"
    print(question1(s, t))
    # Testcase3 - Expected False because Anagram does not exists
    s = "udacity"
    t = "udacious"
    print(question1(s, t))
    # Testcase4 (Edge) - Expected False because substring is longer than string
    s = "city"
    t = "udacity"
    print(question1(s, t))
    # Testcase5 (Edge) - Expected False because string is empty
    s = ""
    t = "udacity"
    print(question1(s, t))

if __name__ == '__main__':
    main()