import json

person = {
    "name" : "alice",
    "age" : 30,
    "addres": {
        "street" : "dunajska0",
        "city" : "ljubljana"
    },
    "married": False 
}

with open(r'C:/Users/dolenec/Desktop/šola/5 semester/TELEKOMUNIKACIJSKI PROTOKOLI/VAJA_1/VAJE/VAJE-3/DATA/person.json', 'w') as f:
    json.dump(person, f, indent=4, sort_keys=True)