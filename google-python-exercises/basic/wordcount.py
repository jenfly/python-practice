#!/usr/bin/python -tt
# Copyright 2010 Google Inc.
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

# Google's Python Class
# http://code.google.com/edu/languages/google-python-class/

"""Wordcount exercise
Google's Python class

The main() below is already defined and complete. It calls print_words()
and print_top() functions which you write.

1. For the --count flag, implement a print_words(filename) function that counts
how often each word appears in the text and prints:
word1 count1
word2 count2
...

Print the above list in order sorted by word (python will sort punctuation to
come before letters -- that's fine). Store all the words as lowercase,
so 'The' and 'the' count as the same word.

2. For the --topcount flag, implement a print_top(filename) which is similar
to print_words() but which prints just the top 20 most common words sorted
so the most common word is first, then the next most common, and so on.

Use str.split() (no arguments) to split on all whitespace.

Workflow: don't build the whole program at once. Get it to an intermediate
milestone and print your data structure and sys.exit(0).
When that's working, try for the next milestone.

Optional: define a helper function to avoid code duplication inside
print_words() and print_top().

"""

import sys

# +++your code here+++
# Define print_words(filename) and print_top(filename) functions.
# You could write a helper utility function that reads a file
# and builds and returns a word/count dict for it.
# Then print_words() and print_top() can just call the utility function.

def remove_val(the_list, val):
  '''Returns a list with all occurrences of val removed'''
  return [value for value in the_list if value != val]

def parse_quotation(string):
  '''Removes quotation marks from beginning/end of string but keeps
  them in the middle for contractions.'''

  s = string
  puncts = ["'", '"', '`'] 
   
  for p in puncts:
    if s.startswith(p): s = s[1:]
    if s.endswith(p):  s = s[:-1]
  
  return s    

def parse_words(line):
  '''Return a list of words in a line, all lowercase'''
  
  # Replace punctuation with spaces
  puncts = [',', '.', ';', ':', '!', '?', '-', '(', ')', '[', ']', '\n', '*']
  for p in puncts:
    line = line.replace(p,' ') 
    
  # Split line into a list of words
  strings = line.split()

  # Parse out quotation marks except for contractions
  strings = [parse_quotation(s) for s in strings]
  
  # Remove '' in case it snuck in there
  strings = remove_val(strings, '')
  
  # Convert all words to lowercase
  strings = [s.lower() for s in strings]
  
  return strings
  

def file_words(filename):
  '''Return dictionary of words and counts in file filename'''

  # Initialize empty dictionary of words in files
  words = {}
  
  # The special mode 'rU' is the "Universal" option for text files where it's 
  # smart about converting different line-endings so they always come through 
  # as a simple '\n'. 
  f = open(filename, 'rU')
  
  # Iterate over the lines of the file
  for line in f:      
      
    # Parse line into list of words 
    strings = parse_words(line)
 
    # For each word, add to word count dictionary
    for s in strings:
      if s not in words.keys():
        words[s] = 1
      else:
        words[s] +=1
    
  f.close()
  return words    

def print_words(filename):
  words = file_words(filename)
  for w in sorted(words):
    print(str(w) + ' ' + str(words[w]))
  return

def print_top(filename):
  '''Print a sorted list of the 20 most common words in the file'''
  ntop = 20
  words = file_words(filename)
  sortlist = sorted(words, key=words.get, reverse=True)
  for i in range(ntop):
    w = sortlist[i]
    print w, words[w]
  return
    
###

# This basic command line argument parsing code is provided and
# calls the print_words() and print_top() functions which you must define.
def main():

  print 'argv[0]', sys.argv[0]
  print 'argv[1]', sys.argv[1]
  print 'argv[2]', sys.argv[2]
  
  if len(sys.argv) != 3:
    print 'usage: ./wordcount.py {--count | --topcount} file'
    sys.exit(1)

  option = sys.argv[1]
  filename = sys.argv[2]
  if option == '--count':
    print_words(filename)
  elif option == '--topcount':
    print_top(filename)
  else:
    print 'unknown option: ' + option
    sys.exit(1)

if __name__ == '__main__':
  main()
