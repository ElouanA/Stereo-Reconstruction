Pour faire tourner l'ensemble des fonctions il faut ouvrir un invite de commande avec l'interpréteur Python et executer les scripts en se plaçant dans le même répertoire que le script. 

Le script TestInit.py permet de réaliser la calibration à partir des paires d'images placées dans le répertoire du script et nommées: "D (index).jpg". On peut changer le nom des photos à ouvrir dans le script (dans les fonctions cv2.imread()). 
Il permet également d'effectuer la reconstruction 3d à partir d'une paire d'image d'indice index.
On  change le nombre de paires de calibration ainsi que l'indice de la paire dont on va reconstruire le modèle 3D dans les paramètres globaux du script. 
C'est ce script qui regroupe l'ensemble des mes fonctions. Les commentaires sur l'explication des différentes fonctions se trouvent dans ce script. Le script CameraFeed.py regroupe les mêmes fonctionnalités mais appliquées directement à l'image renvoyée par une camera Stereo et non pas sur des photos enregistrées dans le répertoire courant. 

Le script PairePhoto.py permet de prendre des paires de photos stéréo avec la caméra. Lance le script et on appuie sur "p" pour prendre une paire. La paire de photo est sauvegardée dans le répertoire courant. 

Le script CameraFeed.py permet de réaliser la calibration et la reconstruction directement à l'aide des caméras. Dans un premier temps on prend des photos du patern de calibration (un exemple de patern est présent dans le dossier) en lancant le script en en appuyant sur "p" lorsque le patern est visible sur les deux images. 
Il faut prendre une 20-aine de photos du patern de calibration, puis appuyer sur "q". 
La calibration ne semble pas fonctionner systématiquement à cause du flou de la caméra lorsque l'on bouge. Il faudrait trouver une solution pour résoudre ce problème. Il faut éventuellement réaliser la calibration à part, vérifier que la calibration est parfaite puis sauvegarder l'ensemble des données de calibration.
Une fois la calibration effectuée, on peut voir la paire stéréo rectifiée, on vérifie que la calibration s'effectue bien en montrant les lignes épipolaires, elles sont parallèles et de même coordonée Y si la calibration est précise. 
On voit ensuite la paire stéréo rectifiée ainsi que la carte de disparité. En appuyant sur "P" on exporte le nuage de point calculé à partir des données de calibration et de la carte de disparité qui s'affiche à l'écran. 

ICP.py est une implémentation de l'algorithme iterative closest point qui permet de faire correspondre deux nuages de points par une transformation rigide. Cette implémentation n'est pas très efficace, et de plus ce n'est pas exactement ce que l'on veut réaliser puisque dans cet algorithme, la correspondance entre les nuages de points est supposée.
Or nous cherchons à faire correspondre les nuages de points de reconstruction entre les instants successifs donc il y a des features du nuages de points qui sont nouveaux et manquants à l'instant t+dt. 
Pour réaliser cette tâche il faudrait regarder du côté de cette librairie mais je n'ai pas réussi à faire marcher les fonctions en python, les exemples OpenCv de cette bilbiothèques sont tous en C++ ou C#: 
https://github.com/opencv/opencv_contrib/tree/master/modules/surface_matching
https://docs.opencv.org/3.0-beta/modules/surface_matching/doc/surface_matching.html

Le script Stereo.py réalise seulement la tache de calculer la carte de disparité à partir d'une paire stéréo rectifiée. L'utilisation de se script permet de jouer plus facilement avec les paramètres des fonctions Stereo. 

