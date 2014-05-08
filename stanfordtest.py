import nlpio,json

if __name__ == '__main__':
    print json.dumps(nlpio.stanfordParse('The world is so pretty.'),indent=4)
    print json.dumps(nlpio.stanfordParse('I like trains. They are nice and clean, almost as clean as their tracks.'),indent=4)
