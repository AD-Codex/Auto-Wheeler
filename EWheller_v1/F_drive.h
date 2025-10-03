
#define F_INIT1 8
#define F_INIT2 9
#define F_PWM   10



void init_FDrive() {
  pinMode(  F_INIT1, OUTPUT);
  pinMode(  F_INIT2, OUTPUT);
  pinMode(  F_PWM  , OUTPUT);

  digitalWrite( F_INIT1, LOW);
  digitalWrite( F_INIT2, LOW);
  digitalWrite( F_PWM, LOW);
  
}


int forward_drive( int linear_x){

  if ( linear_x >0) {
    digitalWrite( F_INIT1, HIGH);
    digitalWrite( F_INIT2, LOW);
    
    int PWM_linear_x = linear_x*255/200;

    analogWrite( F_PWM, 50+ PWM_linear_x);
  }
  else {
    digitalWrite( F_INIT1, LOW);
    digitalWrite( F_INIT2, LOW);
    digitalWrite( F_PWM, LOW);
  }
  
  

  
}
