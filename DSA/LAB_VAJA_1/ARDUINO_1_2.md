void setup() {
  pinMode(12,OUTPUT);
  pinMode(10,OUTPUT);
  pinMode(11,OUTPUT);
}

void posljipodatek(byte podatek){
  digitalWrite(12, LOW);    
  digitalWrite(10, LOW);    
  digitalWrite(11, LOW);  

    for (int i = 0; i < 8; i++){
      if(podatek & B00000001){
        digitalWrite(12, HIGH);
      }
      else{
        digitalWrite(12, LOW);
      }
      digitalWrite(10, HIGH);
      digitalWrite(10, LOW);
      podatek = podatek >> 1;
      }

  digitalWrite(11, HIGH);    
  digitalWrite(11, LOW);    
  delay(100);
}

void loop() {
  /*
  digitalWrite(12, LOW);    
  digitalWrite(10, LOW);    
  digitalWrite(11, LOW);    

  digitalWrite(12, LOW);    
  digitalWrite(10, HIGH);    
  digitalWrite(10, LOW);    

  byte podatek = B01010101;

    for (int i = 0; i < 8; i++){
      if(podatek & B00000001){
        digitalWrite(12, HIGH);
      }
      else{
        digitalWrite(12, LOW);
      }
      digitalWrite(10, HIGH);
      digitalWrite(10, LOW);
      podatek = podatek >> 1;
      }

    

    digitalWrite(11, HIGH);    
    digitalWrite(11, LOW);    
    delay(100);
*/
posljipodatek(B00000000);
  delay(100);
posljipodatek(B00000001);
  delay(100);
posljipodatek(B00000011);
  delay(100);
posljipodatek(B00000111);
  delay(100);
posljipodatek(B00001111);
  delay(100);
posljipodatek(B00011111);
  delay(100);
posljipodatek(B00111111);
  delay(100);
posljipodatek(B01111111);
  delay(100);
posljipodatek(B11111111);

}
