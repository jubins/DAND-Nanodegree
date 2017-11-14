"""
QUESTION#5
Find the element in a singly linked list that's m elements from the end. For example, if a linked list has 5 elements,
the 3rd element from the end is the 3rd element. The function definition should look like question5(ll, m), where ll
is the first node of a linked list and m is the "mth number from the end". You should copy/paste the Node class below
to use as a representation of a node in the linked list. Return the value of the node at that position.
"""
class Node(object):
  def __init__(self, data):
    """Initializing a node"""
    self.data = data
    self.next = None

class LinkedList:
    def __init__(self, head=None):
        """Initializes the linkedlist"""
        self.head = head

    def insertAtBeginning(self, data):
        """Inserts data at begining of the linkedlist"""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def reverse(self):
        """Reverses the LinkedList"""
        prev_node = None
        curr_node = self.head
        while (curr_node is not None):
            next = curr_node.next
            curr_node.next = prev_node
            prev_node = curr_node
            curr_node = next
        self.head = prev_node

    def traverse(self, m):
        """Traverses the LinkedList"""
        curr_node = self.head
        count = 1 #since count starts from 0
        while(curr_node is not None):
            if count == m:
                return curr_node.data
            curr_node = curr_node.next
            count += 1
        return


def question5(ll, m):
    """Traverses after reversing the linkedlist"""
    return ll.traverse(m)


def main():
    ll = LinkedList()
    ll.insertAtBeginning(10)
    ll.insertAtBeginning(20)
    ll.insertAtBeginning(30)
    ll.insertAtBeginning(40)
    ll.insertAtBeginning(50)
    ll.reverse()

    print(question5(ll, 1)) #querying 1st element from end
    print(question5(ll, 3)) #querying 3rd element from end
    print(question5(ll, 2)) #querying 2nd element from end
    print(question5(ll, 5)) #querying 5th element from end


if __name__ == '__main__':
    main()