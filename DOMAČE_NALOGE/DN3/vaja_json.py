import json

with open(r'C:\Users\dolenec\Desktop\šola\5 semester\TELEKOMUNIKACIJSKI PROTOKOLI\VAJA_1\DOMAČE_NALOGE\DN3\DATA\zacetniData.json', 'r') as f:
    zacetniData = json.load(f)

with open(r'C:\Users\dolenec\Desktop\šola\5 semester\TELEKOMUNIKACIJSKI PROTOKOLI\VAJA_1\DOMAČE_NALOGE\DN3\DATA\updateData.json', 'r') as f:
    updateData = json.load(f)

zacetniSlovar = {person['name']: {'age': person['age'], 'married': person['married'], 'employed': person['employed']} for person in zacetniData['persons']}

#update_data = [x for x in updateData['persons'] if x != 'name']




for update_person in updateData['persons']:
    name = update_person['name']
    if name in zacetniSlovar:
        updated_info = {key: value for key, value in update_person.items() if key != 'name'}
        zacetniSlovar[name].update(updated_info)




updated_persons_list = []

for name, person_info in zacetniSlovar.items():
    person = {'name': name}
    person.update(person_info)  
    updated_persons_list.append(person)

updated_data = {'persons': updated_persons_list}


with open(r'C:\Users\dolenec\Desktop\šola\5 semester\TELEKOMUNIKACIJSKI PROTOKOLI\VAJA_1\DOMAČE_NALOGE\DN3\DATA\novaDatoteka.json', 'w') as f:
    json.dump(updated_data, f, indent=4, sort_keys=True, ensure_ascii=False)
