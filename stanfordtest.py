import nlpio,json

if __name__ == '__main__':
    print json.dumps(nlpio.stanfordParse('The world is so pretty.'),indent=4)
    print json.dumps(nlpio.stanfordParse('I like trains. They are nice and clean, almost as clean as their tracks.'),indent=2)
    print json.dumps(nlpio.stanfordParse('Jim worked at a hotel. He was very happy with his life until Jane came into it. She was very arrogant and he disliked her. Still today, he\'s not very happy.'),indent=2)
