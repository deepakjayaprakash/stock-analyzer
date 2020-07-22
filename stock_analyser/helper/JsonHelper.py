import datetime
import json


def dump_json(input):
    def myconverter(o):
        if isinstance(o, datetime.datetime):
            return o.__str__()

    return json.dumps(input, default=myconverter)
