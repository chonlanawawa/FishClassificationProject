import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model_dir = 'finalmodel/model.h5'
model = model=tf.keras.models.load_model(model_dir)

class_names = ['Abalistes_stellatus', 'Abudefduf_saxatilis', 'Acanthemblemaria_spinosa', 'Acanthochromis_polyacanthus', 'Acanthurus_achilles', 'Acanthurus_chirurgus', 'Acanthurus_coeruleus', 'Acanthurus_dussumieri', 'Acanthurus_japonicus', 'Acanthurus_leucosternon', 'Acanthurus_lineatus', 'Acanthurus_maculiceps', 'Acanthurus_nigricans', 'Acanthurus_nigrofuscus', 'Acanthurus_nigroris', 'Acanthurus_olivaceus', 'Acanthurus_pyroferus', 'Acanthurus_sohal', 'Acanthurus_tennenti', 'Acanthurus_thompsoni', 'Acanthurus_triostegus', 'Acanthurus_tristis', 'Acanthurus_xanthopterus', 'Aetobatus_narinari', 'Alectis_indicus', 'Amblyapistus_taenianotus', 'Amblycirrhitus_pinos', 'Amblyeleotris_diagonalis', 'Amblyeleotris_guttata', 'Amblyeleotris_randalli', 'Amblyeleotris_steinitzi', 'Amblyeleotris_wheeleri', 'Amblyglyphidodon_aureus', 'Amblygobius_decussatus', 'Amblygobius_hectori', 'Amblygobius_phalaena', 'Amblygobius_rainfordi', 'Amphiprion_clarkii', 'Amphiprion_percula+Amphiprion_ocellaris', 'Amphiprion_perideraion', 'Amphiprion_polymnus', 'Amphiprion_sebae', 'Amphirion_frenatus+Amphirion_melanopus', 'Anisotremus_virginicus', 'Antennarius spp', 'Antennarius_striatus', 'Apogon_maculatus', 'Apogon_nigrofasciatus', 'Apolemichthys_griffisi', 'Apolemichthys_xanthurus', 'Aptychotrema_rostrata', 'Archamia_zosterophora', 'Arothron_hispidus', 'Arothron_immaculatus', 'Arothron_manilensis', 'Arothron_mappa', 'Arothron_meleagris', 'Arothron_nigropunctatus', 'Arothron_stellatus', 'Assessor_flavissimus', 'Assessor_macneilli', 'Atelomycterus spp', 'Atrosalarias_fuscus', 'Balistapus_undulatus', 'Balistes_punctatus', 'Balistes_vetula', 'Balistoides_conspicillum', 'Balistoides_viridescens', 'Bodianus_bilunulatus', 'Bodianus_bimaculatus', 'Bodianus_diana', 'Bodianus_mesothorax', 'Bodianus_pulchellus', 'Bodianus_rufus', 'Bothus spp', 'Caesio_xanthonota', 'Calloplesiops_altivelis', 'Cantherhines_dumerili', 'Canthigaster_bennetti', 'Canthigaster_coronata', 'Canthigaster_jactator', 'Canthigaster_papua', 'Canthigaster_rostrata', 'Canthigaster_valentini', 'Carcharhinus_melanopterus', 'Centropyge_acanthops', 'Centropyge_argi', 'Centropyge_aurantonotus', 'Centropyge_bicolor', 'Centropyge_bispinosa', 'Centropyge_deborae', 'Centropyge_eibli', 'Centropyge_ferrugata', 'Centropyge_flavicauda', 'Centropyge_flavissima', 'Centropyge_heraldi', 'Centropyge_loricula', 'Centropyge_multicolor', 'Centropyge_multifasciata', 'Centropyge_potteri', 'Centropyge_tibicen', 'Centropyge_venustus', 'Centropyge_vroliki', 'Cephalopholis_argus', 'Cephalopholis_formosa', 'Cephalopholis_fulva', 'Cephalopholis_miniata', 'Cephalopholis_panamensis', 'Cephalopholis_polleni', 'Cetoscarus_bicolor', 'Chaetodermis_penicilligerus', 'Chaetodon_auriga', 'Chaetodon_capistratus', 'Chaetodon_ephippium', 'Chaetodon_falcula', 'Chaetodon_lunula', 'Chaetodon_mertensii', 'Chaetodon_paucifasciatus', 'Chaetodon_rafflesii', 'Chaetodon_sedentarius', 'Chaetodon_semilarvatus', 'Chaetodon_striatus', 'Chaetodon_tinkeri', 'Chaetodon_ulietensis', 'Chaetodon_unimaculatus', 'Chaetodontoplus_caeruleopunctatus', 'Chaetodontoplus_duboulayi', 'Chaetodontoplus_meridithii', 'Cheilinus_diagrammus', 'Chelmon_rostratus', 'Chiloscyllium_plagiosum', 'Chiloscyllium_punctatum', 'Choerodon_fasciatus', 'Chromis_amboinensis', 'Chromis_cyanea', 'Chromis_iomelas', 'Chromis_limbaughi', 'Chromis_lineata', 'Chromis_margaritifer', 'Chromis_nitida', 'Chromis_retrofasciata', 'Chromis_viridis', 'Chromis_xanthura', 'Chrysiptera_caeruleolineata', 'Chrysiptera_cyanea', 'Chrysiptera_galba', 'Chrysiptera_hemicyanea', 'Chrysiptera_parasema', 'Chrysiptera_rex', 'Chrysiptera_rollandi', 'Chrysiptera_springeri', 'Chrysiptera_starcki', 'Chrysiptera_talboti', 'Chrysiptera_taupou', 'Chrysiptera_tricincta', 'Cirrhilabrus_aurantidorsalis', 'Cirrhilabrus_cyanopleura', 'Cirrhilabrus_exquisitus', 'Cirrhilabrus_filamentosus', 'Cirrhilabrus_flavidorsalis', 'Cirrhilabrus_jordani', 'Cirrhilabrus_laboutei', 'Cirrhilabrus_lineatus', 'Cirrhilabrus_lubbocki', 'Cirrhilabrus_luteovittatus', 'Cirrhilabrus_punctatus', 'Cirrhilabrus_rhomboidalis', 'Cirrhilabrus_rubrisquamis', 'Cirrhilabrus_rubriventralis', 'Cirrhilabrus_ryukyuensis', 'Cirrhilabrus_scottorum', 'Cirrhilabrus_solorensis', 'Cirrhitichthys_aprinus', 'Cirrhitichthys_aureus', 'Cirrhitichthys_falco', 'Cirrhitichthys_oxycephalus', 'Cirrhitichthys_polyactis', 'Cirripectes_stigmaticus', 'Congrogadus_subducens', 'Coris_formosa', 'Coris_gaimard', 'Coryphopterus_glaucofraenum', 'Corythoichthys_haematopterus', 'Cromileptes_altivelis', 'Crossosalarias_macrospilus', 'Cryptocentrus_aurora', 'Cryptocentrus_cinctus', 'Cryptocentrus_pavoninoides', 'Ctenochaetus_binotatus', 'Ctenochaetus_hawaiiensis', 'Ctenochaetus_striatus', 'Ctenochaetus_strigosus', 'Ctenochaetus_tominiensis', 'Ctenogobiops_tangaroai', 'Cypho_purpurascens', 'Dascyllus_albisella', 'Dascyllus_auripinnis', 'Dascyllus_carneus', 'Dascyllus_flavicaudus', 'Dascyllus_marginatus', 'Dascyllus_melanurus+Dascyllus_aruanus', 'Dascyllus_reticulatus', 'Dascyllus_trimaculatus', 'Dendrochirus_barberi', 'Dendrochirus_biocellatus', 'Dendrochirus_brachypterus', 'Dendrochirus_zebra', 'Diademichthys_lineatus', 'Diodon_holocanthus', 'Diodon_hystrix', 'Diplobatis_ommata', 'Dischistodus_prosopotaenia', 'Discordipinna_griessingeri', 'Doryrhamphus_dactyliophorus', 'Doryrhamphus_janssi', 'Doryrhamphus_pessuliferus', 'Echidna_catenata', 'Echidna_nebulosa', 'Echidna_polyzona', 'Ecsenius_bicolor', 'Ecsenius_bimaculatus', 'Ecsenius_gravieri', 'Ecsenius_lineatus', 'Ecsenius_midas', 'Ecsenius_namiyei', 'Ecsenius_stigmatura', 'Elacatinus_multifasciatus', 'Elacatinus_oceanops', 'Elacatinus_puncticulatus', 'Emblemaria_pandionis', 'Enchelycore_pardalis', 'Enchelyurus_flavipes', 'Epinephelus_fasciatus', 'Epinephelus_flavocaeruleus', 'Epinephelus_summana', 'Eucrossorhinus_dasypogon', 'Eviota_pellucida', 'Forcipiger_flavissimus', 'Genicanthus_bellus', 'Genicanthus_semifasciatus', 'Ginglymostoma_cirratum', 'Glaucostegus_typus', 'Gnathanodon_speciosus', 'Gobiodon_acicularis', 'Gobiodon_atrangulatus', 'Gobiodon_citrinus', 'Gobiodon_okinawae', 'Gobioides_broussonnetii', 'Gomphosus_varius', 'Gramma_brasiliensis', 'Gramma_loreto', 'Gramma_melacara', 'Grammistes_sexlineatus', 'Gymnomuraena_zebra', 'Gymnothorax_favagineus', 'Gymnothorax_fimbriatus', 'Gymnothorax_funebris', 'Gymnothorax_kidako', 'Gymnothorax_melatremus', 'Gymnothorax_meleagris', 'Gymnothorax_miliaris', 'Gymnothorax_nudivomer', 'Gymnothorax_saxicola', 'Gymnura spp', 'Halichoeres_chloropterus', 'Halichoeres_chrysus', 'Halichoeres_hortulanus', 'Halichoeres_iridis', 'Halichoeres_melanurus', 'Halichoeres_trispilus', 'Hemiscyllium_ocellatum', 'Hemitrygon_akajei', 'Heniochus_diphreutes', 'Heteroconger_hassi', 'Heterodontus_francisci', 'Heterodontus_portusjacksoni', 'Heteropriacanthus_cruentatus', 'Himantura_uarnak', 'Hippocampus spp', 'Hippocampus_bargibanti', 'Histrio_histrio', 'Holacanthus_bermudensis', 'Holacanthus_ciliaris', 'Holacanthus_passer', 'Holacanthus_tricolor', 'Hoplolatilus_luteus', 'Hoplolatilus_marcosi', 'Hoplolatilus_purpureus', 'Hoplolatilus_starcki', 'Hypanus spp', 'Hypoplectrus_gemma', 'Hypoplectrus_gummigutta', 'Hypoplectrus_guttavarius', 'Hypoplectrus_indigo', 'Hypoplectrus_nigricans', 'Hypoplectrus_puella', 'Hypoplectrus_unicolor', 'Hypsypops_rubicunda', 'Inimicus_didactylus', 'Iracundus_signifer', 'Labracinus_cyclophthalmus', 'Labroides_bicolor', 'Labroides_dimidiatus', 'Labroides_phthirophagus', 'Lactoria_cornuta', 'Lutjanus_adetii', 'Lutjanus_sebae', 'Lythrypnus_dalli', 'Macolor_niger', 'Macropharyngodon_geoffroyi', 'Macropharyngodon_meleagris', 'Malacoctenus_boehlkei', 'Manonichthys_alleni', 'Meiacanthus_bundoon', 'Meiacanthus_grammistes', 'Meiacanthus_mossambicus', 'Meiacanthus_nigrolineatus', 'Meiacanthus_oualanensis', 'Melichthys_indicus', 'Melichthys_niger', 'Melichthys_vidua', 'Meuschenia_hippocrepis', 'Microspathodon_chrysurus', 'Muraena_lentiginosa', 'Mustelus spp', 'Mycteroperca_rosacea', 'Myliobatis spp', 'Myrichthys_colubrinus', 'Myrichthys_maculosus', 'Myripristis_jacobus', 'Myripristis_vittata', 'Narcine spp', 'Naso_lituratus', 'Naso_lopezi', 'Naso_unicornis', 'Naso_vlamingii', 'Nemateleotris_decora', 'Nemateleotris_magnifica', 'Neocirrhitus_armatus', 'Neoglyphidodon_crossi', 'Neoglyphidodon_melas', 'Neoglyphidodon_nigroris', 'Neopomacentrus_azysron', 'Neotrygon_kuhlii', 'Novaculichthys_taeniourus', 'Odonus_niger', 'Ogilbyina_novaehollandiae', 'Ophioblennius_atlanticus', 'Opistognathus_aurifrons', 'Opistognathus_lonchurus', 'Opistognathus_rosenblatti', 'Opistognathus_whitehurstii', 'Orectolobus spp', 'Ostorhinchus_aureus', 'Ostorhinchus_compressus', 'Ostorhinchus_cyanosoma', 'Ostracion_cubicus', 'Ostracion_solorensis', 'Oxycirrhites_typus', 'Oxymonacanthus_longirostris', 'Paracanthurus_hepatus', 'Paracheilinus_carpenteri', 'Paracirrhites_arcatus', 'Paracirrhites_forsteri', 'Paracirrhites_hemistictus', 'Paracirrhites_xanthus', 'Paraglyphidodon_oxyodon', 'Paraluteres_prionurus', 'Paramonacanthus_japonicus', 'Parapterois_heterura', 'Parascorpaena_mossambica', 'Parupeneus_barberinoides', 'Parupeneus_barberinus', 'Parupeneus_cyclostomus', 'Parupeneus_multifasciatus', 'Pastinachus_sephen', 'Pervagor_melanocephalus', 'Pervagor_spilosoma', 'Pholidichthys_leucotaenia', 'Pholidochromis_cerasina', 'Platax_orbicularis', 'Platax_pinnatus', 'Platax_teira', 'Platyrhinoidis_triseriata', 'Plectorhinchus_albovittatus', 'Plectorhinchus_chaetodonoides', 'Plectorhinchus_diagrammus+Plectorhinchus_orientalis', 'Plectorhinchus_lineatus', 'Plectorhinchus_picus', 'Plectropomus_laevis', 'Pogonoperca_punctata', 'Pomacanthus_annularis', 'Pomacanthus_arcuatus', 'Pomacanthus_asfur', 'Pomacanthus_imperator', 'Pomacanthus_maculosus', 'Pomacanthus_navarchus', 'Pomacanthus_paru', 'Pomacanthus_semicirculatus', 'Pomacanthus_xanthometopon', 'Pomacanthus_zonipectus', 'Pomacentrus_alleni', 'Pomacentrus_amboinensis', 'Pomacentrus_auriventris', 'Pomacentrus_bankanensis', 'Pomacentrus_caeruleus', 'Pomacentrus_moluccensis', 'Pomacentrus_nigromarginatus', 'Pomacentrus_pavo', 'Pomacentrus_simsiang', 'Pomacentrus_smithi', 'Pomacentrus_vaiuli', 'Premnas_biaculeatus', 'Priolepis_aureoviridis', 'Priolepis_nocturna', 'Pristigenys_serrula', 'Pseudanthias_bartlettorum', 'Pseudanthias_bicolor', 'Pseudanthias_cooperi', 'Pseudanthias_huchtii', 'Pseudanthias_hypselosoma', 'Pseudanthias_parvirostris', 'Pseudanthias_pleurotaenia', 'Pseudanthias_rubrizonatus', 'Pseudanthias_squamipinnis', 'Pseudobalistes_fuscus', 'Pseudobatos spp', 'Pseudocheilinus_evanidus', 'Pseudocheilinus_hexataenia', 'Pseudocheilinus_ocellatus', 'Pseudocheilinus_octotaenia', 'Pseudocheilinus_tetrataenia', 'Pseudochromis_aldabraensis', 'Pseudochromis_aureus', 'Pseudochromis_bitaeniatus', 'Pseudochromis_coccinicauda', 'Pseudochromis_cyanotaenia', 'Pseudochromis_diadema', 'Pseudochromis_dilectus', 'Pseudochromis_elongatus', 'Pseudochromis_flammicauda', 'Pseudochromis_flavivertex', 'Pseudochromis_fridmani', 'Pseudochromis_fuscus', 'Pseudochromis_paccagnellae', 'Pseudochromis_porphyreus', 'Pseudochromis_sankeyi', 'Pseudochromis_splendens', 'Pseudochromis_springeri', 'Pseudochromis_steenei', 'Pseudochromis_veliferus', 'Pterapogon_kauderni', 'Ptereleotris_evides', 'Ptereleotris_hanae', 'Ptereleotris_zebra', 'Pteroidichthys_amboinensis', 'Pterois spp', 'Pygoplites_diacanthus', 'Rhina_ancylostoma', 'Rhinecanthus_aculeatus', 'Rhinecanthus_assasi', 'Rhinecanthus_rectangulus', 'Rhinecanthus_verrucosus', 'Rhinomuraena_quaesita', 'Rhinopias_aphanes', 'Rhinopias_eschmeyeri+Rhinopias_frondosa', 'Rhinoptera_bonasus', 'Salarias_fasciatus', 'Salarias_ramosus', 'Salarias_segmentatus', 'Sargocentron_tiere', 'Sargocentron_xantherythrum', 'Scartella_cristata', 'Scarus_taeniopterus', 'Scorpaenopsis_macrochir', 'Scorpaenopsis_papuensis', 'Scorpaenopsis_possi', 'Sebastapistes_cyanostigma', 'Selene_vomer', 'Serranus_scriba', 'Serranus_tigrinus', 'Serranus_tortugarum', 'Siganus_doliatus', 'Siganus_guttatus', 'Siganus_magnificus', 'Siganus_unimaculatus', 'Siganus_uspi', 'Siganus_vulpinus', 'Signigobius_biocellatus', 'Soleichthys_heterorhinos', 'Sphaeramia_nematoptera', 'Sphaeramia_orbicularis', 'Sphyrna_tiburo', 'Stegastes_diencaeus', 'Stegastes_planifrons', 'Stegostoma_fasciatum', 'Stonogobiops_dracula', 'Stonogobiops_nematodes', 'Stonogobiops_yasha', 'Sufflamen_albicaudatum', 'Sufflamen_bursa', 'Sufflamen_chrysopterum', 'Symphorichthys_spilurus', 'Synanceja_verrucosa', 'Synchiropus_ocellatus', 'Synchiropus_picturatus', 'Synchiropus_splendidus', 'Synchiropus_stellatus', 'Taenianotus_triacanthus', 'Taeniura_lymma', 'Terapon_jarbua', 'Tetrosomus_gibbosus', 'Thalassoma_bifasciatum', 'Thalassoma_hebraicum', 'Thalassoma_jansenii', 'Thalassoma_lucasanum', 'Thalassoma_lunare', 'Thalassoma_lutescens', 'Thalassoma_quinquevittatum', 'Thalassoma_trilobatum', 'Torpedo spp', 'Triaenodon_obesus', 'Triakis_scyllium', 'Triakis_semifasciata', 'Trimma_cana', 'Trygonoptera_ovalis+Trygonoptera_testacea', 'Trygonorrhina_fasciata', 'Urobatis spp', 'Urolophus_gigas', 'Urotrygon_chilensis', 'Valenciennea_helsdingenii', 'Valenciennea_longipinnis', 'Valenciennea_puellaris', 'Valenciennea_sexguttata', 'Valenciennea_strigata', 'Valenciennea_wardii', 'Variola_louti', 'Xanthichthys_auromarginatus', 'Xanthichthys_caeruleolineatus', 'Xanthichthys_mento', 'Xanthichthys_ringens', 'Zanclus_cornutus', 'Zapteryx_Brevirostris+Zapteryx_exasperata', 'Zapteryx_xyster', 'Zebrasoma_desjardinii', 'Zebrasoma_flavescens', 'Zebrasoma_scopas', 'Zebrasoma_veliferum', 'Zebrasoma_xanthurum']

selected_classes = ['Amphiprion_percula+Amphiprion_ocellaris', 'Chelmon_rostratus', 'Nemateleotris_magnifica', 'Paracanthurus_hepatus']

st.set_page_config(page_title="Aquarium fish species classification", layout="wide")
st.title("Aquarium fish species classification")
st.markdown(f"<h4 style='color: #C0C0C0; font-size: 15px;'>Deployment of an image classification model of aquarium fish species with 549 classes. By Chonlana Kruawuthikun (AI Builders 2024)</h4>", unsafe_allow_html=True)

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = tf.image.resize(img_array, (224, 224))
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image):
    predictions = model(image, training=False)
    return predictions

selected_class = st.selectbox("choose from examples:", ["None selected"] + selected_classes)
if selected_class != "None selected":
    images_container = st.empty()
    with images_container.container():
        if selected_class == "Amphiprion_percula+Amphiprion_ocellaris":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(f'example_classes/{selected_classes[0]}/0.jpg', use_column_width=True)
            with col2:
                st.image(f'example_classes/{selected_classes[0]}/1.jpg', use_column_width=True)
            with col3:
                st.image(f'example_classes/{selected_classes[0]}/2.jpg', use_column_width=True)
        if selected_class == "Chelmon_rostratus":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(f'example_classes/{selected_classes[1]}/0.jpg', use_column_width=True)
            with col2:
                st.image(f'example_classes/{selected_classes[1]}/1.jpg', use_column_width=True)
            with col3:
                st.image(f'example_classes/{selected_classes[1]}/2.jpg', use_column_width=True)
        if selected_class == "Nemateleotris_magnifica":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(f'example_classes/{selected_classes[2]}/0.jpg', use_column_width=True)
            with col2:
                st.image(f'example_classes/{selected_classes[2]}/1.jpg', use_column_width=True)
            with col3:
                st.image(f'example_classes/{selected_classes[2]}/2.jpg', use_column_width=True)
        if selected_class == "Paracanthurus_hepatus":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(f'example_classes/{selected_classes[3]}/0.jpg', use_column_width=True)
            with col2:
                st.image(f'example_classes/{selected_classes[3]}/1.jpg', use_column_width=True)
            with col3:
                st.image(f'example_classes/{selected_classes[3]}/2.jpg', use_column_width=True)

uploaded_file = st.file_uploader("or upload your image here", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with st.columns(3)[1]:
     st.image(image.resize((224, 224)), use_column_width=True)
    
    img_array = preprocess_image(image)
    img_array = tf.cast(img_array, tf.float32)

    predictions = predict_image(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.markdown(f"<h4 style='text-align: center; font-size: 20px;'>The species in the image is {predicted_class} with confidence of {confidence:.2f}%</h4>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2>List of species</h2>", unsafe_allow_html=True)
columns = st.columns(5)
for i, class_name in enumerate(class_names):
    with columns[i % 5]:
        st.markdown(f"<small>{class_name}</small>", unsafe_allow_html=True)

st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #f0f2f6;
    }
    h2 {
        align-items: center;
        justify-content: center;
        text-align: center;
        color: #C0C0C0;
    }
    small {
        font-size: 14px;
        color: #A0A0A0;
    }
    </style>
    """, unsafe_allow_html=True)
