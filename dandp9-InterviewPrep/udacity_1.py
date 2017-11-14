"""
QUESTION#1
Given two strings s and t, determine whether some anagram of t is a substring of s.
For example: if s = "udacity" and t = "ad", then the function returns True. Your function
definition should look like: question1(s, t) and return a boolean True or False.
"""

def permutations(t, l, r, permuted):
    '''helper function for question1'''
    if l==r:
        permuted.append(''.join(t))
    else:
        for i in range(l, r+1):
            t[l], t[i] = t[i], t[l]
            permutations(t, l+1, r, permuted)
            t[l], t[i] = t[i], t[l]

#First I create permutations of string t, store all them in a list
# then I check if any of the permutations exists in the string s
def question1(s, t):
    if s == t:
        return True
    if len(s) == 0 or len(t) == 0:
        return False
    permuted = []
    permutations(list(t), 0, len(t)-1, permuted)
    for i in permuted:
        if i in s:
            return True
    return False

def main():
    s = "udacity"
    t = "ad"
    print(question1(s, t))

    s = "udacity"
    t = "acid"
    print(question1(s, t))

    s = "udacity"
    t = "udacious"
    print(question1(s, t))

    s = "city"
    t = "udacity"
    print(question1(s, t))

if __name__ == '__main__':
    main()