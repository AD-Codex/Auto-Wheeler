
// encode limits/ range 170, -170
// left limit --->  455    -30 degree     -0.523
// center value ->  625      0 degree      0.00
// right limit -->  795     30 degree      0.532


#define S_ANALOG  A0

int S_right_max = 800;
int S_left_min  = 450;


void init_SEncode() {
  pinMode( S_ANALOG, INPUT);
}


int SEnc_read(){
  int En_value = analogRead(S_ANALOG);
  return En_value;
}


int left_limit(){
  if ( SEnc_read() <= S_left_min){
//    Serial.println("left clicked");
    return 1;
  }
  else {
    return 0;
  }
}

int right_limit(){
  if ( SEnc_read() >= S_right_max){
//    Serial.println("right clicked");
    return 1;
  }
  else {
    return 0;
  }
}





// Function to map the ADC value to the desired range
float mapToRange(int value, int fromLow, int fromHigh, float toLow, float toHigh) {
  return ((float)(value - fromLow) / (fromHigh - fromLow)) * (toHigh - toLow) + toLow;
}

float S_angel_read(){
  int En_value = analogRead(S_ANALOG);
  float mapped_value = mapToRange(En_value, S_left_min, S_right_max, -0.532, 0.532);

//  Serial.print("ADC Value: ");
//  Serial.print(En_value);
//  Serial.print("\tMapped Value: ");
//  Serial.println(mapped_value);
  
  return mapped_value;
}
