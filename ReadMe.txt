Pour faire tourner l'ensemble des fonctions il faut ouvrir un invite de commande avec l'interpr�teur Python et executer les scripts en se pla�ant dans le m�me r�pertoire que le script. 

Le script TestInit.py permet de r�aliser la calibration � partir des paires d'images plac�es dans le r�pertoire du script et nomm�es: "D (index).jpg". On peut changer le nom des photos � ouvrir dans le script (dans les fonctions cv2.imread()). 
Il permet �galement d'effectuer la reconstruction 3d � partir d'une paire d'image d'indice index.
On  change le nombre de paires de calibration ainsi que l'indice de la paire dont on va reconstruire le mod�le 3D dans les param�tres globaux du script. 
C'est ce script qui regroupe l'ensemble des mes fonctions. Les commentaires sur l'explication des diff�rentes fonctions se trouvent dans ce script. Le script CameraFeed.py regroupe les m�mes fonctionnalit�s mais appliqu�es directement � l'image renvoy�e par une camera Stereo et non pas sur des photos enregistr�es dans le r�pertoire courant. 

Le script PairePhoto.py permet de prendre des paires de photos st�r�o avec la cam�ra. Lance le script et on appuie sur "p" pour prendre une paire. La paire de photo est sauvegard�e dans le r�pertoire courant. 

Le script CameraFeed.py permet de r�aliser la calibration et la reconstruction directement � l'aide des cam�ras. Dans un premier temps on prend des photos du patern de calibration (un exemple de patern est pr�sent dans le dossier) en lancant le script en en appuyant sur "p" lorsque le patern est visible sur les deux images. 
Il faut prendre une 20-aine de photos du patern de calibration, puis appuyer sur "q". 
La calibration ne semble pas fonctionner syst�matiquement � cause du flou de la cam�ra lorsque l'on bouge. Il faudrait trouver une solution pour r�soudre ce probl�me. Il faut �ventuellement r�aliser la calibration � part, v�rifier que la calibration est parfaite puis sauvegarder l'ensemble des donn�es de calibration.
Une fois la calibration effectu�e, on peut voir la paire st�r�o rectifi�e, on v�rifie que la calibration s'effectue bien en montrant les lignes �pipolaires, elles sont parall�les et de m�me coordon�e Y si la calibration est pr�cise. 
On voit ensuite la paire st�r�o rectifi�e ainsi que la carte de disparit�. En appuyant sur "P" on exporte le nuage de point calcul� � partir des donn�es de calibration et de la carte de disparit� qui s'affiche � l'�cran. 

ICP.py est une impl�mentation de l'algorithme iterative closest point qui permet de faire correspondre deux nuages de points par une transformation rigide. Cette impl�mentation n'est pas tr�s efficace, et de plus ce n'est pas exactement ce que l'on veut r�aliser puisque dans cet algorithme, la correspondance entre les nuages de points est suppos�e.
Or nous cherchons � faire correspondre les nuages de points de reconstruction entre les instants successifs donc il y a des features du nuages de points qui sont nouveaux et manquants � l'instant t+dt. 
Pour r�aliser cette t�che il faudrait regarder du c�t� de cette librairie mais je n'ai pas r�ussi � faire marcher les fonctions en python, les exemples OpenCv de cette bilbioth�ques sont tous en C++ ou C#: 
https://github.com/opencv/opencv_contrib/tree/master/modules/surface_matching
https://docs.opencv.org/3.0-beta/modules/surface_matching/doc/surface_matching.html

Le script Stereo.py r�alise seulement la tache de calculer la carte de disparit� � partir d'une paire st�r�o rectifi�e. L'utilisation de se script permet de jouer plus facilement avec les param�tres des fonctions Stereo. 

