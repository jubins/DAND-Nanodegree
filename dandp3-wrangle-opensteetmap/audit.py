def update(name, mapping): 
    words = name.split()
    for w in range(len(words)):
        if words[w] in mapping:
            if words[w].lower() not in ['suite', 'ste.', 'ste']: 
                # For example, don't update 'Suite E' to 'Suite East'
                words[w] = mapping[words[w]]
                name = " ".join(words)
    return name

def correct_abbreviations(problem_field, tree, mapping):
    for tag in tree.iter('tag'):
        if problem_field in tag.attrib['k']:
            words = tag.attrib['v']
            updated_words = ""
            for word in words.split(" "):
                if (word in mapping):
                    word = mapping[word]
                updated_words = "".join(word)
            tag.attrib['v'] = updated_words



import xml.etree.cElementTree as ET
SAMPLE_FILE = "new-york-sample.osm"
tree = ET.parse(SAMPLE_FILE)

street_mapping = {"St":"Street", "St.":"Street", "Ave": "Avenue", "ave": "Avenue", "Blvd": "Boulevard", "Rd": "Road"}
name_type_mapping = {"Pkwy": "Parkway", "St": "Street", "St.":"Street", "Ave": "Avenue", "ave": "Avenue", "Blvd": "Boulevard", "Rd": "Road"}

correct_abbreviations('street', tree, street_mapping)
correct_abbreviations('name_type', tree, name_type_mapping)
tree.write(SAMPLE_FILE)