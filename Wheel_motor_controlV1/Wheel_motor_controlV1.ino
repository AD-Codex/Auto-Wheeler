char incoming;
int speeds = 0;;

void setup() {
  // put your setup code here, to run once:
  pinMode(3, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(5, OUTPUT);

  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(4,HIGH);
  digitalWrite(5,LOW);

  if (Serial.available() > 0) {
    incoming = Serial.read();
    if ( int(incoming) == 49){
        speeds = 0;
        Serial.print("Speed : ");
        Serial.println(speeds);
    }
    else if ( int(incoming) == 50){
        speeds = 10;
        Serial.print("Speed : ");
        Serial.println(speeds);
    }
    else if ( int(incoming) == 51){
        speeds = 20;
        Serial.print("Speed : ");
        Serial.println(speeds);
    }
    else if ( int(incoming) == 52){
        speeds = 50;
        Serial.print("Speed : ");
        Serial.println(speeds);
    }
    else if ( int(incoming) == 53){
        speeds = 100;
        Serial.print("Speed : ");
        Serial.println(speeds);
    }
    else if ( int(incoming) == 54){
        speeds = 150;
        Serial.print("Speed : ");
        Serial.println(speeds);
    }
    else if ( int(incoming) == 55){
        speeds = 200;
        Serial.print("Speed : ");
        Serial.println(speeds);
    }
    else if ( int(incoming) == 56){
        speeds = 250;
        Serial.print("Speed : ");
        Serial.println(speeds);
    }
//    Serial.println(incoming);
  }

  analogWrite(3,speeds);
  Serial.print("Speed : ");
  Serial.println(speeds);

 delay(10);
}
