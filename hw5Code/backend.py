import os
import sqlite3
import re
from frames import Frame

TIMEPAT = re.compile("\d{1,2}[:]\d{1,2}")
PRICEPAT = re.compile("\d{1,3}[.]\d{1,2}")
REPLACEMENTS = [(" it's ", ' it is '), (" don't ", ' do not '), (" doesn't ", ' does not '), (" didn't ", ' did not '), (" you'd ", ' you would '), (" you're ", ' you are '), (" you'll ", ' you will '), (" i'm ", ' i am '), (" they're ", ' they are '), (" that's ", ' that is '), (" what's ", ' what is '), (" couldn't ", ' could not '), (" i've ", ' i have '), (" we've ", ' we have '), (" can't ", ' cannot '), (" i'd ", ' i would '), (" i'd ", ' i would '), (" aren't ", ' are not '), (" isn't ", ' is not '), (" wasn't ", ' was not '), (" weren't ", ' were not '), (" won't ", ' will not '), (" there's ", ' there is '), (" there're ", ' there are '), (' . . ', ' . '), (' restaurants ', ' restaurant -s '), (' hotels ', ' hotel -s '), (' laptops ', ' laptop -s '), (' cheaper ', ' cheap -er '), (' dinners ', ' dinner -s '), (' lunches ', ' lunch -s '), (' breakfasts ', ' breakfast -s '), (' expensively ', ' expensive -ly '), (' moderately ', ' moderate -ly '), (' cheaply ', ' cheap -ly '), (' prices ', ' price -s '), (' places ', ' place -s '), (' venues ', ' venue -s '), (' ranges ', ' range -s '), (' meals ', ' meal -s '), (' locations ', ' location -s '), (' areas ', ' area -s '), (' policies ', ' policy -s '), (' children ', ' child -s '), (' kids ', ' kid -s '), (' kidfriendly ', ' kid friendly '), (' cards ', ' card -s '), (' upmarket ', ' expensive '), (' inpricey ', ' cheap '), (' inches ', ' inch -s '), (' uses ', ' use -s '), (' dimensions ', ' dimension -s '), (' driverange ', ' drive range '), (' includes ', ' include -s '), (' computers ', ' computer -s '), (' machines ', ' machine -s '), (' families ', ' family -s '), (' ratings ', ' rating -s '), (' constraints ', ' constraint -s '), (' pricerange ', ' price range '), (' batteryrating ', ' battery rating '), (' requirements ', ' requirement -s '), (' drives ', ' drive -s '), (' specifications ', ' specification -s '), (' weightrange ', ' weight range '), (' harddrive ', ' hard drive '), (' batterylife ', ' battery life '), (' businesses ', ' business -s '), (' hours ', ' hour -s '), (' one ', ' 1 '), (' two ', ' 2 '), (' three ', ' 3 '), (' four ', ' 4 '), (' five ', ' 5 '), (' six ', ' 6 '), (' seven ', ' 7 '), (' eight ', ' 8 '), (' nine ', ' 9 '), (' ten ', ' 10 '), (' eleven ', ' 11 '), (' twelve ', ' 12 '), (' anywhere ', ' any where '), (' good bye ', ' goodbye ')]
DOMAINS = ['restaurant', 'hotel', 'attraction', 'train']
EXCLUDED_COLUMNS = ['entrance fee', 'id', 'location', 'takesbookings', 'price', 'trainID', 'introduction']

dir = os.path.dirname(os.path.abspath(__file__))

DBS = {}
for domain in DOMAINS:
    DBS[domain] = sqlite3.connect(os.path.join(dir, 'db', f'{domain}-dbase.db')).cursor()


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text


def normalize(text):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))

    # normalize postcode
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                    text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace time and and price
    # text = re.sub(TIMEPAT, ' [value_time] ', text)
    # text = re.sub(PRICEPAT, ' [value_price] ', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\":\<>@\(\)]', '', text)

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in REPLACEMENTS:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text

def backend(frame):
    """Query the database.

    Parameters:
    - frame (Frame): the frame representing the query
    
    Returns:
    - list of Frames: the query results
    """
    if not isinstance(frame, Frame):
        raise TypeError("frame must be of type Frame")

    domain = ''
    for d in DOMAINS:
        if d in frame.type:
            domain = d

    if not domain:
        return []

    items = frame.args.items()

    # query the db
    sql_query = "select * from {}".format(domain)

    flag = True
    for key, val in items:
        key2 = key.split('-')[-1]
        if key2 == 'arriveby':
            key2 = 'arriveBy'
        elif key2 == 'leaveat':
            key2 = 'leaveAt'

        if val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care" :
            pass
        else:
            if flag:
                sql_query += " where "
                val2 = val.replace("'", "''")
                val2 = normalize(val2)
                if key2 == 'leaveAt':
                    sql_query += r" " + key2 + " > " + r"'" + val2 + r"'"
                elif key2 == 'arriveBy':
                    sql_query += r" " + key2 + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key2 + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                val2 = val.replace("'", "''")
                val2 = normalize(val2)
                if key2 == 'leaveAt':
                    sql_query += r" and " + key2 + " > " + r"'" + val2 + r"'"
                elif key2 == 'arriveBy':
                    sql_query += r" and " + key2 + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key2 + "=" + r"'" + val2 + r"'"

    try:
        results = DBS[domain].execute(sql_query).fetchall()
    except:
        return []

    cols = [c[0] for c in DBS[domain].description]
    cols = [domain + '-' + c.lower() if c.lower() not in EXCLUDED_COLUMNS else None for c in cols]

    output_frames = []
    for result in results:
        args = {
            col: val
            for col, val in zip(cols, result)
            if col is not None
        }
        output_frames.append(Frame(domain, args))

    return output_frames


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: backend.py <frame>\n(If <frame> has spaces, please enclose in quotes.)", file=sys.stderr)
        exit(1)
    frame = Frame.from_str(sys.argv[1])
    results = backend(frame)
    for frame in results:
        print(frame)

