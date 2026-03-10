// Code Tiny Toulouse Robot Race : 3 juin 2023
// Code d'une voiture qui avance et qui compte les tours de cricuits et qui s'arrête une fois fini.

#include <NewPing.h>
#include <ESP8266WiFi.h>
#include <LiquidCrystal_I2C.h>
#include <Servo.h>

#define TRIGGER_PIN 16
#define ECHO_PIN 14
#define MAX_DISTANCE 200 // Distance maximale pour les mesures en centimètres
#define BUTTON_PIN 2

Servo ESC1;

const int SPEED = 60; // Entre 60 et 100

const int targetTours = 5;
int tours = 0;
bool dernierTourCompte = false;

NewPing sonar(TRIGGER_PIN, ECHO_PIN, MAX_DISTANCE);
LiquidCrystal_I2C lcd(0x27,16,2);   // adresse i2c , nombre de colonnes, nombre de lignes de l'écran

void setSpeed(int speed){
  int angle = map(speed, 0, 100, 0, 200); //Sets servo positions to different speeds
  ESC1.write(angle);
  //Serial.println(speed);
}

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN,INPUT);
  ESC1.attach(13);
  Serial.println("Bonjour");
  lcd.init();   // initialisation du LCD
  lcd.backlight();   // active le rétroéclairage lcd.noBacklight() pour le désactiver
  lcd.setCursor(0,0);   // mettre le curseur à la première ligne, première colonne
  lcd.setCursor(0,0);   // mettre le curseur à la deuxième ligne, première colonne
  lcd.print("Tour :");
  lcd.print(0);
}
int delayTime = 0;

bool done = false ;

void loop() {
  Serial.println(digitalRead(BUTTON_PIN));
  if ( done ){
    return ;
  }
  delayTime -= 1;
  unsigned int distance = sonar.ping_cm();
  //Serial.println(distance);
  setSpeed(SPEED);
  if (distance != 0 and delayTime <= 0) {
    if (distance < 90) {
      tours++;
      Serial.print("Tour numero ");
      Serial.println(tours);
      lcd.setCursor(0,0);   
      lcd.print("Tour :");
      lcd.print(tours);
      delayTime = 120;
    }
  }
  if (tours >= targetTours) {
    setSpeed(0);
    Serial.println("Objectif atteint. Arrêt du moteur.");
    lcd.setCursor(0,0);   
    lcd.print("Termine ...");
    done = true ;
  } else {
    // Faire avancer la voiture
    setSpeed(SPEED);
  }
  
}
