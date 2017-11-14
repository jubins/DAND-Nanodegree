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
    for i in range(len(string)-1):
        reversed_string = []
        reverse(string[i:], reversed_string)
        reversed_string = ''.join(reversed_string)
        #print(string[i:], reversed_string)
        if (string[i:] == reversed_string):
            palindrome =  a[i:].strip(' \n\t')
            return palindrome
        else:
            continue
    return ''

def question2(a):
    longest_palindrome = ''
    string = ''.join(x for x in list(a.lower()) if x.isalpha())
    palindrome = palindromize(string)
    if len(longest_palindrome) < len(palindrome):
        longest_palindrome = palindrome

    reversed_string = []
    reverse(string, reversed_string)
    reversed_string = ''.join(reversed_string)
    palindrome = palindromize(reversed_string)
    if len(longest_palindrome) < len(palindrome):
        longest_palindrome = palindrome
    return longest_palindrome



if __name__ == '__main__':
    a = 'That Was it a car or a cat I saw?'
    print(question2(a))

    a = 'RACECAR'
    print(question2(a))

    a = 'ABCBACADCBBCDA'
    print(question2(a))