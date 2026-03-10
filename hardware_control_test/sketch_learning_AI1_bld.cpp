#include <NewPing.h>
#include <ESP8266WiFi.h>
#include <LiquidCrystal_I2C.h>
#include <iostream>
#include <vector>
#include <cmath>

// Définir les dimensions de la piste
const int largeurPiste = 100; // Largeur de la piste en centimètres
const int longueurPiste = 200; // Longueur de la piste en centimètres

// Définir les hyperparamètres
const float tauxApprentissage = 0.1; // Taux d'apprentissage
const float facteurActualisation = 0.9; // Facteur d'actualisation
const int nbIterationsEntrainement = 10000; // Nombre d'itérations d'entraînement
const int nbActions = 4; // Nombre d'actions possibles (avancer, reculer, tourner à gauche, tourner à droite)

// Tableau de valeurs Q pour les états et les actions
std::vector<std::vector<float>> QTable;

// Fonction pour initialiser la table de valeurs Q
void initialiserQTable() {
    QTable.resize(largeurPiste, std::vector<float>(longueurPiste, 0.0));
}

// Fonction pour choisir une action en fonction de l'état actuel
int choisirAction(int etat) {
    // Générer un nombre aléatoire entre 0 et 1
    float randValue = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    // Exploration : choisir une action aléatoire avec une probabilité epsilon
    if (randValue < epsilon) {
        return rand() % nbActions;
    }
    // Exploitation : choisir l'action avec la valeur Q maximale
    else {
        float maxQValue = -std::numeric_limits<float>::infinity();
        int meilleureAction = 0;

        for (int action = 0; action < nbActions; action++) {
            if (QTable[etat][action] > maxQValue) {
                maxQValue = QTable[etat][action];
                meilleureAction = action;
            }
        }

        return meilleureAction;
    }
}

// Fonction pour mettre à jour les valeurs Q en fonction de la récompense et du nouvel état
void mettreAJourQTable(int ancienEtat, int action, float recompense, int nouvelEtat) {
    float ancienneValeur = QTable[ancienEtat][action];
    float nouvelleValeur = ancienneValeur + tauxApprentissage * (recompense + facteurActualisation * max(QTable[nouvelEtat]) - ancienneValeur);
    QTable[ancienEtat][action] = nouvelleValeur;
}

int main() {
    // Initialiser la table de valeurs Q
    initialiserQTable();

    // Boucle d'entraînement
    for (int iteration = 0; iteration < nbIterationsEntrainement; iteration++) {
        // Initialiser l'état initial de la voiture

        while (distance > 0.05) {
            mesurerEtat();  // Mesurer l'état actuel de la voiture en utilisant les capteurs

            int actafair = choisirAction(int etat);  // Choisir une action en fonction de l'état actuel

            executerAction(int actafair)  // Exécuter l'action sur la voiture

            calculerRecompense(int resultatAction)// Mesurer la récompense en fonction du résultat de l'action

            mettreAJourValeursQ(int etatActuel, int action, float recompense, int nouvelEtat) // Mettre à jour les valeurs Q en fonction de la récompense et du nouvel état
        }
    }

    // Après l'entraînement, la voiture peut utiliser les valeurs Q pour prendre des décisions lors de la course réelle

    return 0;
}

// Fonction pour mesurer l'état actuel de la voiture en utilisant les capteurs
int mesurerEtat() {
    // Mesurer la distance du lidar
    float distanceLidar = mesurerDistanceLidar();

    // Mesurer l'état des capteurs infrarouges
    bool capteurInfrarougeGauche = mesurerEtatCapteurInfrarougeGauche();
    bool capteurInfrarougeCentre = mesurerEtatCapteurInfrarougeCentre();
    bool capteurInfrarougeDroite = mesurerEtatCapteurInfrarougeDroite();

    // Mesurer la distance des capteurs ultrasons latéraux
    float distanceUltrasonGauche = mesurerDistanceUltrasonGauche();
    float distanceUltrasonDroite = mesurerDistanceUltrasonDroite();

    // Convertir les mesures en valeurs discrètes
    int etat = 0;
    if (distanceLidar < 50) {
        etat += 1;
    }
    if (capteurInfrarougeGauche) {
        etat += 2;
    }
    if (capteurInfrarougeCentre) {
        etat += 4;
    }
    if (capteurInfrarougeDroite) {
        etat += 8;
    }
    if (distanceUltrasonGauche < 30) {
        etat += 16;
    }
    if (distanceUltrasonDroite < 30) {
        etat += 32;
    }

    return etat;
}

// Fonction pour choisir une action en fonction de l'état actuel
int choisirAction(int etat) {
    // Définir les actions possibles
    const int AVANCER = 0;
    const int TOURNER_GAUCHE = 1;
    const int TOURNER_DROITE = 2;
    // Ajoutez d'autres actions selon vos besoins

    // Effectuer une action en fonction de l'état actuel
    int action;
    if (etat == 0) {
        action = AVANCER;
    } else if (etat == 1) {
        action = TOURNER_GAUCHE;
    } else if (etat == 2) {
        action = TOURNER_DROITE;
    } else {
        // Action par défaut si l'état n'a pas de correspondance
        action = AVANCER;
    }

    return action;
}

// Fonction pour exécuter une action sur la voiture
void executerAction(int action) {
    // Effectuer l'action correspondante
    if (action == AVANCER) {
        // Code pour faire avancer la voiture
    } else if (action == TOURNER_GAUCHE) {
        // Code pour tourner la voiture vers la gauche
    } else if (action == TOURNER_DROITE) {
        // Code pour tourner la voiture vers la droite
    } else {
        // Action par défaut ou action non reconnue
        // Ajoutez le code correspondant à votre logique
    }
}


// Fonction pour mesurer la récompense en fonction du résultat de l'action
float calculerRecompense(int resultatAction) {
    float recompense = 0.0;

    // Mesurer le résultat de l'action et attribuer une récompense
    if (resultatAction == SUCCES) {
        // Action réussie, récompense positive
        recompense = 1.0;
    } else if (resultatAction == ECHEC) {
        // Action échouée, récompense négative
        recompense = -1.0;
    } else {
        // Autre résultat ou action non reconnue
        // Ajoutez le code correspondant à votre logique
    }

    return recompense;
}


// Fonction pour mettre à jour les valeurs Q en fonction de la récompense et du nouvel état
void mettreAJourValeursQ(int etatActuel, int action, float recompense, int nouvelEtat) {
    // Obtenir la valeur Q actuelle pour l'état et l'action
    float ancienneValeur = QTable[etatActuel][action];

    // Trouver la meilleure valeur Q pour le nouvel état
    float meilleureValeur = getMaxValeurQ(nouvelEtat);

    // Mettre à jour la valeur Q pour l'état et l'action actuels
    float nouvelleValeur = ancienneValeur + tauxApprentissage * (recompense + facteurActualisation * meilleureValeur - ancienneValeur);
    QTable[etatActuel][action] = nouvelleValeur;
}
