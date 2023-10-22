import json

with open(r'C:/Users/dolenec/Desktop/šola/5 semester/TELEKOMUNIKACIJSKI PROTOKOLI/VAJA_1/VAJE/VAJE-3/DATA/person.json', 'r') as f:
    deserialized_person = json.load(f)

print(deserialized_person)

deserialized_person['age'] = 40
deserialized_person['married'] = False

with open(r'C:/Users/dolenec/Desktop/šola/5 semester/TELEKOMUNIKACIJSKI PROTOKOLI/VAJA_1/VAJE/VAJE-3/DATA/person.json', 'w') as f:
    json.dump(deserialized_person, f, indent=4, sort_keys=True)