void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
  pinMode(2,OUTPUT);
}

double getTemp(int adc) //funkicjaa AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
{
      return ((((adc/1023.00)*5000-500)/10)-4);
    }

// the loop routine runs over and over again forever:
void loop() {
  // read the input on analog pin 0:
  int sensorValueA0 = analogRead(A0);
  int sensorValueA1 = analogRead(A1);

  // print out the value you read:
  //Serial.println(sensorValueA0);
  delay(500);  // delay in between reads for stability

if(sensorValueA0 > 1000 ){
        digitalWrite(2, HIGH);
      }
      else{
        digitalWrite(2, LOW);
      }

  //Serial.println((((sensorValueA1/1023.00)*5000-500)/10)-4);
  //Serial.println(sensorValueA1);

double temperatura = getTemp(sensorValueA1);

Serial.println(temperatura);
}

