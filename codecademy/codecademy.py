print("Hello. I love cats...")



# -----------------------------------------------
# Strings

# Single or double quotation marks:
str1 = "Hello kittens"
str2 = 'Hello kittens'

# Escape characters
str3 = 'This isn\'t flying, this is falling with style!'

# Indexing letters
fifth_letter = "MONTY"[4]

# String methods
parrot = "Norwegian Blue"
print(len(parrot))
print(parrot.lower())
print(parrot.upper())
pi = 3.14
print(str(pi))

# Concatenation
print("Spam " + "and " + "eggs")
print("The value of pi is around " + str(3.14))

# -----------------------------------------------
# Formatted printing

meal = 44.50
tax = 0.0675
tip = 0.15

meal = meal + meal * tax
total = meal + meal*tip

print("%.2f" % total)

string_1 = "Camelot"
string_2 = "place"

print("Let's not go to %s. 'Tis a silly %s." % (string_1, string_2))

# -----------------------------------------------
# Defining numeric arrays:
x1 = range(5) # 0, 1, 2, 3, 4   
x2 = range(0, 10) # 0, 1, 2, 3, 4
x3 = range(0,10,2) # 0, 2, 4, 6, 8

# -----------------------------------------------
# Two methods for iterating over a list

animals = ["hamster", "rabbit", "cat", "gerbil"]

print("--- Loop over items:")
for item in animals:
	print("Hello " + item)

print("--- Loop over indices:")
for i in range(len(animals)):
    print(str(i) + ") " + animals[i])

# -----------------------------------------------
# User input

#name = raw_input("What is your name?")
#quest = raw_input("What is your quest?")
#color = raw_input("What is your favorite color?")

#print("Ah, so your name is %s, your quest is %s, " \
#"and your favorite color is %s." % (name, quest, color)

# -----------------------------------------------
# Date and Time

from datetime import datetime
now = datetime.now()

print('%s/%s/%s %s:%s:%s' % (now.month, now.day, now.year, now.hour, now.minute, now.second))


# -----------------------------------------------
# Boolean operators

# Use boolean expressions as appropriate on the lines below!

# Make me false!
bool_one = (2 <= 2) and "Alpha" == "Bravo"  # We did this one for you!

# Make me true!
bool_two = (1<2) or (2<1)

# Make me false!
bool_three = "cat"=="kittens"

# Make me true!
bool_four = not (10>100)

# Make me true!
bool_five = (10>5) and (1<2)

# -----------------------------------------------
# Control Flow: If-Else

def greater_less_equal_5(answer):
    if answer>5:
        return 1
    elif answer<5:          
        return -1
    else:
        return 0
        
print(greater_less_equal_5(4))       

# -----------------------------------------------
# Pyg-Latin Translator

def pyg_latin(input_word):
    pyg = 'ay'
    original = input_word
    word = original.lower()
    first = word[0]
    new_word = word[1:len(word)] + first + pyg

    if len(original) > 0 and original.isalpha():
        return new_word
    else:
        return 'empty'

print(pyg_latin('Kittens'))
print(pyg_latin('123Kittens!'))

# -----------------------------------------------
# Defining functions

def cube(number):
    return number**3
    
def by_three(number):
    if number%3 == 0:
        return cube(number)
    else:
        return False   

# -----------------------------------------------
# Built-in functions

import math            # Imports the math module
everything = dir(math) # Sets everything to a list of things from math
print(everything)       # Prints 'em all!

print(max(10, -3, 2.5, 200.1))
print(min(5.1, 3.2, 1))
print(abs(-42))
print(type(43))
print(type(1.2))
print(type('cat'))
print(math.sqrt(13689))

def distance_from_zero(num):
    if type(num) == int or type(num) == float:
        return abs(num)
    else:
        return 'Nope'
        
print(distance_from_zero('kittens'))

# -----------------------------------------------
# Vacation cost

def hotel_cost(nights):
    return 140 * nights

def plane_ride_cost(city):
    if city == 'Charlotte':
        return 183
    elif city == 'Tampa':
        return 220
    elif city == 'Pittsburgh':
        return 222
    elif city == 'Los Angeles':
        return 475
    else:
        return 'Oops'

def rental_car_cost(days):
    cost = 40*days
    if days >= 7:
        cost -= 50
    elif days >= 3:
        cost -= 20
    return cost

def trip_cost(city, days, spending_money):
    return hotel_cost(days) + plane_ride_cost(city) + rental_car_cost(days) + spending_money

print trip_cost('Los Angeles', 5, 600)

# ----------------------------------------------
# Lists

zoo_animals = ["pangolin", "cassowary", "sloth", "zebra"];
# One animal is missing!

if len(zoo_animals) > 3:
	print "The first animal at the zoo is the " + zoo_animals[0]
	print "The second animal at the zoo is the " + zoo_animals[1]
	print "The third animal at the zoo is the " + zoo_animals[2]
	print "The fourth animal at the zoo is the " + zoo_animals[3]
# ----------------------------------------------
suitcase = [] 
suitcase.append("sunglasses")
suitcase.append("bikini")
suitcase.append("flipflops")
suitcase.append("sunscreen")
list_length = len(suitcase) # Set this to the length of suitcase
print "There are %d items in the suitcase." % (list_length)
print suitcase
# ----------------------------------------------
suitcase = ["sunglasses", "hat", "passport", "laptop", "suit", "shoes"]

first  = suitcase[0:2]  # The first and second items (index zero and one)
middle = suitcase[2:4]               # Third and fourth items (index two and three)
last   = suitcase[4:6]               # The last two items (index four and five)
# ----------------------------------------------
animals = "catdogfrog"
cat  = animals[:3]   # The first three characters of animals
dog  = animals[3:6]              # The fourth through sixth characters
frog = animals[6:]              # From the seventh character to the end
# ----------------------------------------------	
animals = ["aardvark", "badger", "duck", "emu", "fennec fox"]
duck_index =  animals.index("duck")  # Use index() to find "duck"
animals.insert(duck_index, "cobra")
print animals # Observe what prints after the insert operation
# ----------------------------------------------             
my_list = [1,9,3,8,5,7]

for number in my_list:
    # Your code here
    print 2 * number
# ----------------------------------------------
start_list = [5, 3, 1, 2, 4]
square_list = []

for number in start_list:
    square_list.append(number ** 2)
square_list.sort()

print square_list
# ----------------------------------------------

# Dictionaries
# ----------------------------------------------
# Assigning a dictionary with three key-value pairs to residents:
residents = {'Puffin' : 104, 'Sloth' : 105, 'Burmese Python' : 106}

print residents['Puffin'] # Prints Puffin's room number
print residents['Sloth']
print residents['Burmese Python']
# ----------------------------------------------
menu = {} # Empty dictionary
menu['Chicken Alfredo'] = 14.50 # Adding new key-value pair
print menu['Chicken Alfredo']
menu['Spaghetti'] = 10
menu['Chocolate Cake'] = 4.5
menu['Lasagne'] = 12.5
print "There are " + str(len(menu)) + " items on the menu."
print menu
# ----------------------------------------------
# key - animal_name : value - location 
zoo_animals = { 'Unicorn' : 'Cotton Candy House',
'Sloth' : 'Rainforest Exhibit',
'Bengal Tiger' : 'Jungle House',
'Atlantic Puffin' : 'Arctic Exhibit',
'Rockhopper Penguin' : 'Arctic Exhibit'}
# A dictionary (or list) declaration may break across multiple lines

# Removing the 'Unicorn' entry. (Unicorns are incredibly expensive.)
del zoo_animals['Unicorn']


del zoo_animals['Sloth']
del zoo_animals['Bengal Tiger']
zoo_animals['Rockhopper Penguin'] = 'Penguin House'

print zoo_animals
# ----------------------------------------------
backpack = ['xylophone', 'dagger', 'tent', 'bread loaf']
backpack.remove('dagger')
# ----------------------------------------------
inventory = {
    'gold' : 500,
    'pouch' : ['flint', 'twine', 'gemstone'], # Assigned a new list to 'pouch' key
    'backpack' : ['xylophone','dagger', 'bedroll','bread loaf'],
    'pocket' : ['seashell', 'strange berry', 'lint']
}

# Adding a key 'burlap bag' and assigning a list to it
inventory['burlap bag'] = ['apple', 'small ruby', 'three-toed sloth']

# Sorting the list found under the key 'pouch'
inventory['pouch'].sort() 

# Your code here
inventory['backpack'].sort()
inventory['backpack'].remove('dagger')
inventory['gold'] += 50
# ----------------------------------------------
def fizz_count(x):
    count = 0
    for item in x:
        if item == 'fizz':
            count += 1
    return count

print fizz_count(['fizz', 'cat', 'kittens', 'fizz'])
# ----------------------------------------------
for letter in "Codecademy":
    print letter
    
# Empty lines to make the output pretty
print
print

word = "Programming is fun!"

for letter in word:
    # Only print out the letter i
    if letter == "i":
        print letter
# ----------------------------------------------
shopping_list = ["banana", "orange", "apple"]

stock = {
    "banana": 6,
    "apple": 0,
    "orange": 32,
    "pear": 15
}
    
prices = {
    "banana": 4,
    "apple": 2,
    "orange": 1.5,
    "pear": 3
}

def compute_bill(food):
    total = 0
    for item in food:
        if stock[item] > 0:
            total += prices[item]
            stock[item] -= 1
    return total
# ----------------------------------------------
lloyd = {
    "name": "Lloyd",
    "homework": [90.0, 97.0, 75.0, 92.0],
    "quizzes": [88.0, 40.0, 94.0],
    "tests": [75.0, 90.0]
}
alice = {
    "name": "Alice",
    "homework": [100.0, 92.0, 98.0, 100.0],
    "quizzes": [82.0, 83.0, 91.0],
    "tests": [89.0, 97.0]
}
tyler = {
    "name": "Tyler",
    "homework": [0.0, 87.0, 75.0, 22.0],
    "quizzes": [0.0, 75.0, 78.0],
    "tests": [100.0, 100.0]
}

def average(numbers):
    total = sum(numbers)
    total = float(total)
    total = total / len(numbers)
    return total

def get_average(student):
    homework = average(student["homework"])
    quizzes = average(student["quizzes"])
    tests = average(student["tests"])
    return 0.1*homework + 0.3*quizzes + 0.6*tests

def get_letter_grade(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"

def get_class_average(students):
    results = [];
    for student in students:
        print student["name"]
        results.append(get_average(student))
    print "results "
    print results
    return average(results)

students = [lloyd, alice, tyler];
avg = get_class_average(students)
print avg
print get_letter_grade(avg)
# ----------------------------------------------
numbers = [3, 5, 7]

def total(numbers):
    result = 0
    for n in numbers:
        result += n
    return result

print(total(numbers))    
# ----------------------------------------------
n = [[1, 2, 3], [4, 5, 6, 7, 8, 9]]

def flatten(lists):
    result = []
    for lst in lists:
        for item in lst:
            result.append(item)
    return result

print flatten(n)
# ----------------------------------------------
# Battleship
from random import randint

board = []

for x in range(5):
    board.append(["O"] * 5)

def print_board(board):
    for row in board:
        print " ".join(row)

print "Let's play Battleship!"
print_board(board)

def random_row(board):
    return randint(0, len(board) - 1)

def random_col(board):
    return randint(0, len(board[0]) - 1)

ship_row = random_row(board)
ship_col = random_col(board)
print ship_row
print ship_col

# Everything from here on should go in your for loop!
# Be sure to indent four spaces!

for turn in range(4):
    print "Turn", turn + 1
    guess_row = int(raw_input("Guess Row:"))
    guess_col = int(raw_input("Guess Col:"))

    if guess_row == ship_row and guess_col == ship_col:
        print "Congratulations! You sunk my battleship!"
        break
    else:
        if (guess_row < 0 or guess_row > 4) or (guess_col < 0 or guess_col > 4):
            print "Oops, that's not even in the ocean."
        elif(board[guess_row][guess_col] == "X"):
            print "You guessed that one already."
        else:
            print "You missed my battleship!"
            board[guess_row][guess_col] = "X"
        print_board(board)
        if turn == 3: print "Game Over"

        

# ----------------------------------------------
# While / else

'''
Something completely different about Python is the while/else construction. while/else is similar to if/else, but there is a difference: the else block will execute anytime the loop condition is evaluated to False. This means that it will execute if the loop is never entered or if the loop exits normally. If the loop exits as the result of a break, the else will not be executed.

In this example, the loop will break if a 5 is generated, and the else will not execute. Otherwise, after 3 numbers are generated, the loop condition will become false and the else will execute.
'''

import random

print "Lucky Numbers! 3 numbers will be generated."
print "If one of them is a '5', you lose!"

count = 0
while count < 3:
    num = random.randint(1, 6)
    print num
    if num == 5:
        print "Sorry, you lose!"
        break
    count += 1
else:
    print "You win!"

# ----------------------------------------------

from random import randint

# Generates a number from 1 through 10 inclusive
random_number = randint(1, 10)

guesses_left = 3
# Start your game!

while guesses_left > 0:
    guess = int(raw_input("Your guess: "))
    if guess == random_number:
        print 'You win!'
        break
    guesses_left -= 1
else:
    print 'You lose'
    
# ---------------------------------------------- 
'''
Multiple lists
It's also common to need to iterate over two lists at once. This is where the built-in zip function comes in handy.

zip will create pairs of elements when passed two lists, and will stop at the end of the shorter list.

zip can handle three or more lists as well!
'''


list_a = [3, 9, 17, 15, 19]
list_b = [2, 4, 8, 10, 30, 40, 50, 60, 70, 80, 90]

for a, b in zip(list_a, list_b):
    # Add your code here!
    if a > b: print a
    else: print b
    
# ----------------------------------------------
'''
For / else
Just like with while, for loops may have an else associated with them.

In this case, the else statement is executed after the for, but only if the for ends normally—that is, not with a break. This code will break when it hits 'tomato', so the else block won't be executed.
'''
fruits = ['banana', 'apple', 'orange', 'tomato', 'pear', 'grape']

print 'You have...'
for f in fruits:
    if f == 'tomato':
        print 'A tomato is not a fruit!' # (It actually is.)
        break
    print 'A', f
else:
    print 'A fine selection of fruits!'
    
   
# ----------------------------------------------
# Sum digits of a number

def digit_sum(n):
    s = str(n)
    result = 0
    for i in range(len(s)):
        result += int(s[i])
    return result

print digit_sum(1234)
print digit_sum(222)

# ----------------------------------------------
# Check for prime number

def is_prime(x):
    if x < 2:
        prime = False
    else:
        prime = True
    for n in range(2, x):
        if x % n == 0:
            prime = False
            break
    return prime

print is_prime(11)
print is_prime(4)
print is_prime(35)

# ----------------------------------------------
# Reverse a string

def reverse(text):
    rev = ''
    ind = -1
    while ind >= -len(text):
        rev += text[ind]
        ind -= 1
    return rev
    
print reverse('Fuzzy kitten')

# ----------------------------------------------
# Anti-vowel

def anti_vowel(text):
    vowel = ['a', 'e', 'i','o','u']
    result = text
    for v in vowel:
        result = result.replace(v, '')
        result = result.replace(v.upper(), '')
    return result

print(anti_vowel('Hey You!'))

# ----------------------------------------------


    
