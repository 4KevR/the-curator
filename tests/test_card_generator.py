import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from card_generator import read_pdf, create_anki_cards

if __name__ == "__main__":
    path = 'card_generator/test.pdf' # 6 pages, the last page is blank
    text = read_pdf(path)
    cards = create_anki_cards(text)
    print(cards)

'''
You will get output similar to the following.
I checked and only the first question of item 4 has an error.

{
1: [
    {'question': "What is KIT's affiliation?", 
     'answer': 'The Research University in the Helmholtz Association'},
     
    {'question': 'What field of study does the introduction cover?', 
     'answer': 'Natural Language Processing'},
     
    {'question': 'What is the title of the first part introduced in the content?', 
     'answer': '01 - Introduction'}
],

2: [
    {'question': 'Who is the contact for Prof. Jan Niehues at the Institute for Anthropomatics and Robotics?', 
     'answer': 'jan.niehues@kit.edu'},
     
    {'question': 'Where can you find more information about the AI4LT Institute?', 
     'answer': 'http://ai4lt.anthropomatik.kit.edu'},
     
    {'question': 'What is the location of the Practical Labs for AI4LT?', 
     'answer': 'Building 50.20, Room 148'}
],

3: [
    {'question': "Where can I find Jan Niehues' lectures in English?", 
     'answer': 'lecture-translator.kit.edu'},
     
    {'question': 'What is available for each lecture through the Lecture Translation Archive?', 
     'answer': 'English Transcript, Translation into German, and Lecture Recording'},
     
    {'question': 'Which languages are supported by the Ilias Q&A forum?', 
     'answer': 'German and English'}
],

4: [
    {'question': 'What is the topic of discussion on October 2?', 
     'answer': 'Words.'},
     
    {'question': 'What is scheduled for practical sessions on November 14 and 28?', 
     'answer': 'Practical Sessions - Word Representation (November 14) & Sequence Classification & Labeling (November 28).'},
     
    {'question': 'What is the final topic discussed in the course, scheduled for February 25?', 
     'answer': 'Recap.'}
],

5: [
    {'question': 'Who is Jan Niehues, specifically in relation to AI4LT?', 
     'answer': 'Natural Language Processing - Jan Niehues, Institute for Anthropomatics and Robotics.'},
     
    {'question': 'What is suggested as a method to apply knowledge in this context?', 
     'answer': 'Directly apply knowledge.'},
     
    {'question': 'What should participants bring to the event with them?', 
     'answer': 'Bring your own laptop.'}
]
}
'''