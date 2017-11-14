"""
QUESTION#2
Given a string a, find the longest palindromic substring contained in a.
Your function definition should look like question2(a), and return a string.
"""

def reverse(s, r):
    '''helper function for question2 to reverse a string'''
    if len(s)==0:
        return ''
    r.append(s[-1])
    return reverse(s[:-1], r)

def palindromize(string):
    '''helper function for question2 to find palindrome'''
    s = string
    string = ''.join(x for x in list(s.lower()) if x.isalpha())
    for i in range(len(string)-1):
        reversed_string = []
        reverse(string[i:], reversed_string)
        reversed_string = ''.join(reversed_string)
        #print(string[i:], reversed_string)
        if (string[i:] == reversed_string):
            palindrome =  s[i:].strip(' \n\t')
            return palindrome
        else:
            continue
    return ''

def question2(a):
    if len(a) <= 1:
        return a
    longest_palindrome = ''
    string = a#''.join(x for x in list(a.lower()) if x.isalpha())
    #checking palindrome from left to right
    palindrome = palindromize(string)
    if len(longest_palindrome) < len(palindrome):
        longest_palindrome = palindrome

    # checking palindrome from right to left
    reversed_string = []
    reverse(string, reversed_string)
    reversed_string = ''.join(reversed_string)
    palindrome = palindromize(reversed_string)
    if len(longest_palindrome) < len(palindrome):
        longest_palindrome = palindrome

    # checking palindrome in middle

    return longest_palindrome

def main():
    #Testcase1 - Expected: Was it a car or a cat I saw?
    a = 'That Was it a car or a cat I saw?'
    print(question2(a))
    # Testcase2 - Expected: RACECAR
    a = 'RACECAR'
    print(question2(a))
    # Testcase3 - Expected: ADCBBCDA
    a = 'ABCBACADCBBCDA'
    print(question2(a))
    # Testcase4 - Expected: null as no palindrome
    a = 'iPhone'
    print(question2(a))
    # Testcase5 - Expected: A as single character
    a = 'A'
    print(question2(a))


if __name__ == '__main__':
    main()