import xml.etree.ElementTree as ET

tree = ET.parse(r'C:\Users\dolenec\Desktop\šola\5 semester\TELEKOMUNIKACIJSKI PROTOKOLI\VAJA_1\VAJE\VAJE-3\DATA/POMOC/person.xml')
root = tree.getroot()

for elem in root:
    print(f"Element: {elem.tag}, Text: {elem.text}")
4