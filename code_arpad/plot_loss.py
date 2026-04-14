import re
import matplotlib.pyplot as plt

def extract_loss_from_log(file_path):
    trainlosses = []
    validlosses = []
    # Ce pattern cherche "training loss=" suivi d'un nombre entier ou décimal
    loss_pattern = re.compile(r"train loss :  ([0-9]*\.?[0-9]+)")
    validloss_pattern = re.compile(r"Returning ([0-9]*\.?[0-9]+)")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # On cherche toutes les occurrences de loss dans la ligne
            found = loss_pattern.findall(line)
            found_valid = validloss_pattern.findall(line)
            if found_valid:
                validlosses.append(float(found_valid[-1]))
            if found:
                # On convertit en float et on ajoute à notre liste
                # Note : on prend la dernière valeur trouvée sur la ligne si tqdm a print plusieurs fois
                trainlosses.append(float(found[-1]))

    return trainlosses, validlosses



# --- Exécution ---
log_file = '/home/aschaeff/ml-pfa/log_Arpad/test_timing.log'  # Remplace par ton nom de fichier
data_loss = extract_loss_from_log(log_file)

if data_loss:
    plt.figure(figsize=(10, 5))
    plt.plot(data_loss[0], label='Training Loss', color='blue', alpha=0.7)
    plt.plot(data_loss[1], label='Validation Loss', color='orange', alpha=0.7)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Evolution of Training and Validation Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('training_loss_plot.png', dpi=150, bbox_inches='tight')  # Sauvegarde du graphique
    plt.close()
else:
    print("Aucune valeur de loss n'a été trouvée. Vérifie le format du fichier.")