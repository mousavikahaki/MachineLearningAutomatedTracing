#
# Created on 8/22/2019
#
# @author Seyed
#
# Email: mousavikahaki@gmail.com
#

list = ['5', '53','923']
print(list)

for i in list:
    print(i)

# Sorting list based on values
list.sort(reverse=True)
print(list)

''.join(list)


# Sorting list based on len
list.sort(key=len)
print(list)

# concate it
''.join(list)

# based on values
list = [3, 7, 97, 34]
list.sort()
print(list)

# based on values
list = ['3', '5', '27', '34']
list.sort(key=int)
print(list)

# based on first number
list = ['3', '5', '27', '34']
list.sort()
print(list)







# List Of Strings -----------------------------------------------------------
listOfStrings = ['hi', 'hello', 'at', 'this', 'there', 'from']

print(listOfStrings)

'''
Sort List of string alphabetically
'''
listOfStrings.sort()

# Print the list
print(listOfStrings)

'''
Sort List of string alphabetically in Reverse Order
'''
listOfStrings.sort(reverse=True)

print(listOfStrings)

'''
Sort List of string by Length by using len() as custom key function 
'''
listOfStrings.sort(key=len)

print(listOfStrings)

'''
Sort List of string by Numeric Order
'''
listOfNum = ['55', '101', '152', '98', '233', '40', '67']

# It will sort in alphabetical order
listOfNum.sort()

print(listOfNum)

'''
Sort in Ascending numeric order, pass key function that should convert string to integer i.e using int()
'''
listOfNum.sort(key=int)

print(listOfNum)

'''
Sort in Descending numeric order, pass reverse flag along with key function
'''

listOfNum.sort(reverse=True, key=int)

print(listOfNum)
# ----------------------------------------------------------------------

##########################################################  Quick Sort
# This function takes last element as pivot, places
# the pivot element at its correct position in sorted
# array, and places all smaller (smaller than pivot)
# to left of pivot and all greater elements to right
# of pivot
def partition(arr, low, high):
    i = (low - 1)  # index of smaller element
    pivot = arr[high]  # pivot

    for j in range(low, high):

        # If current element is smaller than or
        # equal to pivot
        if arr[j] <= pivot:
            # increment index of smaller element
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return (i + 1)


# The main function that implements QuickSort
# arr[] --> Array to be sorted,
# low  --> Starting index,
# high  --> Ending index

# Function to do Quick sort
def quickSort(arr, low, high):
    if low < high:
        # pi is partitioning index, arr[p] is now
        # at right place
        pi = partition(arr, low, high)

        # Separately sort elements before
        # partition and after partition
        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)



# Driver code to test above

# arr = [10, 7, 8, 9, 1, 5]
arr = ['10', '7', '8', '9', '1', '5']
for i in range(0, len(arr)):
    arr[i] = int(arr[i])

arr = ['a', 'bb', 'a', 'v', 'c', 'ccc']
n = len(arr)
quickSort(arr,0,n-1)
print ("Sorted array is:")
print(arr)

