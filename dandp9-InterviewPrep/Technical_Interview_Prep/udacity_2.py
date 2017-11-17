"""
QUESTION#2
Given a string a, find the longest palindromic substring contained in a.
Your function definition should look like question2(a), and return a string.
"""

def longest_palindrome(a, left_idx, right_idx):
    # find the longest palindrome if centered at idx.
    # idx can be in between elements.
    # left_idx and right_idx are the left and the right element of idx
    l = left_idx
    r = right_idx
    while l >= 0 and r < len(a):
        if a[l] == a[r]:
            l -= 1
            r += 1
        else:
            return l, r
    return l, r


def question2(a):
    # make sure a is a string
    if type(a) != str:
        return "Error: a not string!"

    # make sure a has at least 2 characters
    if len(a) < 2:
        return a

    # check all possible center of palindrome
    pal_left = 0
    pal_right = 1
    for i in range(len(a) - 1):
        # check palindrome centered at i
        l, r = longest_palindrome(a, i, i)
        if r - l - 1 > pal_right - pal_left:
            pal_right = r
            pal_left = l + 1

        # check palindrome centered between i and i+1
        l, r = longest_palindrome(a, i, i + 1)
        if r - l - 1 > pal_right - pal_left:
            pal_right = r
            pal_left = l + 1
    return a[pal_left:pal_right]

def main():
    #Testcase1 - Expected: aba
    a = 'abacdfgdcaba'
    print(question2(a))
    # Testcase2 - Expected: RACECAR
    a = 'RACECAR'
    print(question2(a))
    # Testcase3 - Expected: ADCBBCDA
    a = 'ABCBACADCBBCDA'
    print(question2(a))
    # Testcase4 (Edge) - Expected: null as no palindrome
    a = 'babcbabcbaccba'
    print(question2(a))
    # Testcase5 (Edge) - Expected: A as single character
    a = 'A'
    print(question2(a))


if __name__ == '__main__':
    main()