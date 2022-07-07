#!/bin/bash

mkdir -p images
mkdir -p ML/ml_dis_rf/results

# UNCOMPRESSING KEGG-ORTHOLOGS
bizp2 -d clr_tables/*

wget -q http://cmprod1.cibio.unitn.it/curatedMetagenomicDataAnalyses/sex_u_kegg.tsv.bz2 -O - | bunzip2 -qc > relative_abundances/sex_u_kegg.tsv
wget -q http://cmprod1.cibio.unitn.it/curatedMetagenomicDataAnalyses/age-kegg_clr.tsv.bz2 -O - | bunzip2 -qc > clr_tables/age-kegg_clr.tsv
wget -q http://cmprod1.cibio.unitn.it/curatedMetagenomicDataAnalyses/bmi-kegg_clr.tsv.bz2 -O - | bunzip2 -qc > clr_tables/bmi-kegg_clr.tsv
wget -q http://cmprod1.cibio.unitn.it/curatedMetagenomicDataAnalyses/sex-kegg_clr.tsv.bz2 -O - | bunzip2 -qc > clr_tables/sex-kegg_clr.tsv

## SECTION 1: SPECIES, GENUS, PWY, and KOs META-ANALYSIS ON SEX, AGE, and BMI
python ../python_tools/metaanalyze.py clr_tables/sex-species_clr.tsv -z s__ -re -cc "0.0:1.0" --formula "gender + age + BMI + number_reads" 
python ../python_tools/metaanalyze.py clr_tables/sex-genera_clr.tsv -z g__ -re -cc "0.0:1.0" --formula "gender + age + BMI + number_reads"  
python ../python_tools/metaanalyze.py clr_tables/sex-pathways_clr.tsv -z PWY -re -cc "0.0:1.0" --formula "gender + age + BMI + number_reads"   
python ../python_tools/metaanalyze.py clr_tables/sex-kegg_clr.tsv -z K -re -cc "0.0:1.0" --formula "gender + age + BMI + number_reads" -si dataset_name

python ../python_tools/metaanalyze.py clr_tables/age-species_clr.tsv -z s__ -mc --formula "age + BMI + C(gender) + number_reads"
python ../python_tools/metaanalyze.py clr_tables/age-genera_clr.tsv -z g__ -mc --formula "age + BMI + C(gender) + number_reads"
python ../python_tools/metaanalyze.py clr_tables/age-pathways_clr.tsv -z PWY -mc --formula "age + BMI + C(gender) + number_reads"
python ../python_tools/metaanalyze.py clr_tables/age-kegg_clr_substitution.tsv -z K -mc --formula "age + BMI + C(gender) + number_reads" #-si dataset_name

python ../python_tools/metaanalyze.py clr_tables/bmi-species_clr.tsv -z s__ -mc --formula "BMI + age + C(gender) + number_reads"
python ../python_tools/metaanalyze.py clr_tables/bmi-genera_clr.tsv -z g__ -mc --formula "BMI + age + C(gender) + number_reads"
python ../python_tools/metaanalyze.py clr_tables/bmi-pathways_clr.tsv -z PWY -mc --formula "BMI + age + C(gender) + number_reads"
python ../python_tools/metaanalyze.py clr_tables/bmi-kegg_clr_substitution.tsv -z K -mc --formula "BMI + age + C(gender) + number_reads" #-si dataset_name
 
python ../python_tools/draw_figure_with_ma.py clr_tables/sex-species_clr_metaanalysis.tsv clr_tables/sex-genera_clr_metaanalysis.tsv --names SPC GN --outfile images/SPC_FIG2_meta_5precent_prevalence --how joint_top     -ps "Sex: male" -ns "Sex: female" --x_axis "Standardized mean difference" --y_axis "Microbial taxa" --title "Meta-analysis of sex-related microbial taxa" --a_single 0.2 --a_random 0.05 -re "RE_Effect"     -ci "RE_conf_int" -rq "RE_Effect_Qvalue" -es "_Effect" -qs "_Qvalue" -cbl black -cr darkgoldenrod -cb dodgerblue -il 0.20 --neg_max_rho 0.8 --pos_max_rho 0.8 -a ../paper/latest/datas/sex_u_species.tsv ../paper/latest/datas/sex_u_genera.tsv --narrowed --boxes --imp 30 --check_genera -ms 2

python ../python_tools/draw_figure_with_ma.py clr_tables/sex-pathways_clr_metaanalysis.tsv --outfile images/PWY_FIG2_meta_supp1 --how first -ps "Sex: male" -ns "Sex: female" --x_axis "Standardized mean difference" --y_axis "Metabolic pathway" --title "Meta-analysis of sex-related metabolic pathways" --a_single 0.2 --a_random 0.05 -re "RE_Effect" -ci "RE_conf_int" -rq "RE_Effect_Qvalue" -es "_Effect" -qs "_Qvalue" -cbl black -cr darkgoldenrod -cb dodgerblue -il 0.20 --neg_max_rho 0.6 --pos_max_rho 0.6 -a ../python_vignettes/the_last_time_I_do_this/sex__u_pathways_with_alpha_diversities.tsv --narrowed --boxes --imp 30 -ms 2
  
python ../python_tools/draw_figure_with_ma.py clr_tables/age-species_clr_metaanalysis.tsv clr_tables/age-genera_clr_metaanalysis.tsv --names SPC GN --outfile images/SPC_FIG3_meta_1_5precent_prevalence --how joint_top -ps "Older age" -ns "Younger age" --x_axis "Partial correlation with age" --y_axis "Microbial taxa" --title "Meta-analysis of age-associated microbial taxa" --a_single 0.2 --a_random 0.05 -re "RE_Correlation" -ci "RE_conf_int" -rq "RE_Correlation_Qvalue" -es "_Correlation" -qs "_Qvalue" -cbl black -cr darkgoldenrod -cb dodgerblue -il 0.10 --neg_max_rho 0.4 --pos_max_rho 0.6 -a ../paper/latest/datas/age_u_species.tsv ../paper/latest/datas/age_u_genera.tsv --narrowed --boxes --imp 30 --check_genera
   
python ../python_tools/draw_figure_with_ma.py clr_tables/bmi-species_clr_metaanalysis.tsv clr_tables/bmi-genera_clr_metaanalysis.tsv --names SPC GN --outfile images/SPC_FIG3_meta_2_5precent_prevalence --how joint_top -ps "Higher BMI" -ns "Lower BMI" --x_axis "Partial correlation with BMI" --y_axis "Microbial taxa" --title "Meta-analysis of BMI-associated microbial taxa" --a_single 0.2 --a_random 0.05 -re "RE_Correlation" -ci "RE_conf_int" -rq "RE_Correlation_Qvalue" -es "_Correlation" -qs "_Qvalue" -cbl black -cr darkgoldenrod -cb dodgerblue -il 0.10 --neg_max_rho 0.4 --pos_max_rho 0.4 -a ../paper/latest/datas/bmi_u_species.tsv ../paper/latest/datas/bmi_u_genera.tsv --narrowed --boxes --imp 30 --check_genera
   
../python_tools/draw_figure_with_ma.py clr_tables/age-pathways_clr_metaanalysis.tsv --outfile images/PWY_FIG3_meta_supp1 --how first -ps "Older age associated" -ns "Younger age associated" --x_axis "Partial correlation with age (yrs.)" --y_axis "Metabolic pathway" --title "Meta-analysis of age-associated metabolic pathways" --a_single 0.2 --a_random 0.05 -re "RE_Correlation" -ci "RE_conf_int" -rq "RE_Correlation_Qvalue" -es "_Correlation" -qs "_Qvalue" -cbl black -cr darkgoldenrod -cb dodgerblue -il 0.10 --neg_max_rho 0.4 --pos_max_rho 0.4 -a ../python_vignettes/the_last_time_I_do_this/ageing_data_pathways_touse_with_alpha_diversities.tsv --narrowed --boxes --imp 30 -ms 2
   
../python_tools/draw_figure_with_ma.py clr_tables/bmi-pathways_clr_metaanalysis.tsv --outfile images/PWY_FIG3_meta_supp2 --how first -ps "Higher BMI associated" -ns "Lower BMI associated" --x_axis "Partial correlation with BMI (kg/m^2)" --y_axis "Metabolic pathway" --title "Meta-analysis of BMI-associated metabolic pathways" --a_single 0.2 --a_random 0.05 -re "RE_Correlation" -ci "RE_conf_int" -rq "RE_Correlation_Qvalue" -es "_Correlation" -qs "_Qvalue" -cbl black -cr darkgoldenrod -cb dodgerblue -il 0.10 --neg_max_rho 0.4 --pos_max_rho 0.4 -a ../python_vignettes/the_last_time_I_do_this/BMI_data_pathways_touse_with_alpha_diversities.tsv --narrowed --boxes --imp 30 -ms 2

python ../python_tools/draw_figure_with_ma.py clr_tables/sex-kegg_clr_metaanalysis.tsv --outfile images/KOS_FIG2_meta_supp2 --how first -ps "Sex: male" -ns "Sex: female" --x_axis "Standardized mean difference" --y_axis "" --title "Meta-analysis of sex-related KOs" --a_single 0.2 --a_random 0.05 -re "RE_Effect" -ci "RE_conf_int" -rq "RE_Effect_Qvalue" -es "_Effect" -qs "_Qvalue" -cbl black -cr darkgoldenrod -cb dodgerblue -il 0.20 --neg_max_rho 0.6 --pos_max_rho 0.6 -a ../python_vignettes/the_last_time_I_do_this/sex_KO__usable_with_alpha_diversities.tsv --narrowed --boxes --imp 30 -ms 2
  
#clr_tables/age-kegg_clr_substitution.tsv  clr_tables/bmi-kegg_clr_substitution.tsv
python ../python_tools/draw_figure_with_ma.py clr_tables/age-kegg_clr_substitution_metaanalysis.tsv --outfile images/KOS_FIG3_meta_supp3 --how first -ps "Older age associated" -ns "Younger age associated" --x_axis "Partial correlation with age (yrs.)" --y_axis "" --title "Meta-analysis of age-associated KOs" --a_single 0.2 --a_random 0.05 -re "RE_Correlation" -ci "RE_conf_int" -rq "RE_Correlation_Qvalue" -es "_Correlation" -qs "_Qvalue" -cbl black -cr darkgoldenrod -cb dodgerblue -il 0.10 --neg_max_rho 0.4 --pos_max_rho 0.4 -a ../python_vignettes/the_last_time_I_do_this/age_data_KO_touse_with_alpha_diversities.tsv --narrowed --boxes --imp 30 -ms 2
     
python ../python_tools/draw_figure_with_ma.py clr_tables/bmi-kegg_clr_substitution_metaanalysis.tsv --outfile images/KOS_FIG3_meta_supp4 --how first -ps "Higher BMI associated" -ns "Lower BMI associated" --x_axis "Partial correlation with BMI (kg/m^2)" --y_axis "" --title "Meta-analysis of BMI-associated KOs" --a_single 0.2 --a_random 0.05 -re "RE_Correlation" -ci "RE_conf_int" -rq "RE_Correlation_Qvalue" -es "_Correlation" -qs "_Qvalue" -cbl black -cr darkgoldenrod -cb dodgerblue -il 0.10 --neg_max_rho 0.4 --pos_max_rho 0.4 -a ../python_vignettes/the_last_time_I_do_this/bmi_data_KO_touse_with_alpha_diversities.tsv --narrowed --boxes --imp 30 -ms 2

## SECTION 2: HIERARCHICAL META-ANALYSIS ON 12 DISEASES
python ../python_tools/metaanalyze.py nested_clr_tables/BD_species_clr.tsv          -z s__ --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:BD -sre -H PM
python ../python_tools/metaanalyze.py nested_clr_tables/CRC_pathways_clr.tsv       -z PWY --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:positive -re -H PM

python ../python_tools/metaanalyze.py nested_clr_tables/T2D_species_clr.tsv  -z s__ --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:T2D -re -H PM
python ../python_tools/metaanalyze.py nested_clr_tables/ACVD_pathways_clr.tsv    -z PWY --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:positive -sre -H PM

python ../python_tools/metaanalyze.py nested_clr_tables/CRC_species_clr.tsv  -z s__      --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:CRC -re -H PM
python ../python_tools/metaanalyze.py nested_clr_tables/schizofrenia_pathways_clr.tsv  -z PWY --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:positive -sre -H PM

python ../python_tools/metaanalyze.py nested_clr_tables/ACVD_species_clr.tsv     -z s__ --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:ACVD -sre -H PM
python ../python_tools/metaanalyze.py nested_clr_tables/CD_pathways_clr.tsv       -z PWY  --formula "disease_subtype + C(gender) + age + BMI + number_reads" -cc control:positive -re -H PM

python ../python_tools/metaanalyze.py nested_clr_tables/schizofrenia_species_clr.tsv   -z s__ --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:schizofrenia -sre -H PM
python ../python_tools/metaanalyze.py nested_clr_tables/UC_pathways_clr.tsv -z PWY --formula "disease_subtype + C(gender) + age + BMI + number_reads" -cc control:positive -re -H PM

python ../python_tools/metaanalyze.py nested_clr_tables/CD_species_clr.tsv          -z s__ --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:IBD -re -H PM
python ../python_tools/metaanalyze.py nested_clr_tables/MECFS_pathways_clr.tsv     -z PWY --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:positive -sre -H PM

python ../python_tools/metaanalyze.py nested_clr_tables/UC_species_clr.tsv -z s__ --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:IBD -re -H PM
python ../python_tools/metaanalyze.py nested_clr_tables/asthma_pathways_clr.tsv  -z PWY --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:positive -sre -H PM

python ../python_tools/metaanalyze.py nested_clr_tables/MECFS_species_clr.tsv    -z s__   --formula "study_condition + C(gender) + age + BMI + number_reads" -cc "control:ME/CFS" -sre -H PM
python ../python_tools/metaanalyze.py nested_clr_tables/STH_pathways_clr.tsv -z PWY --formula "study_condition + C(gender) + age + BMI + number_reads" -cc neg:positive -sre -H PM

python ../python_tools/metaanalyze.py nested_clr_tables/asthma_species_clr.tsv   -z s__ --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:asthma -sre -H PM
python ../python_tools/metaanalyze.py nested_clr_tables/cirrhosis_pathways_clr.tsv  -z PWY --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:positive -sre -H PM

python ../python_tools/metaanalyze.py nested_clr_tables/STH_species_clr.tsv  -z s__ --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:STH -sre -H PM
python ../python_tools/metaanalyze.py nested_clr_tables/cirrhosis_species_clr.tsv -z s__  --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:cirrhosis -sre -H PM

python ../python_tools/metaanalyze.py nested_clr_tables/migraine_pathways_clr.tsv  -z PWY --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:positive -sre -H PM
python ../python_tools/metaanalyze.py nested_clr_tables/BD_pathways_clr.tsv      -z PWY --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:positive -sre -H PM

python ../python_tools/metaanalyze.py nested_clr_tables/migraine_species_clr.tsv   -z s__ --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:migraine -sre -H PM
python ../python_tools/metaanalyze.py nested_clr_tables/T2D_pathways_clr.tsv -z PWY --formula "study_condition + C(gender) + age + BMI + number_reads" -cc control:positive -re -H PM

python ../python_tools/hierarchical_metaanalysis.py \
    --names CRC T2D CD UC ACVD cirrhosis schizofrenia asthma STH BD "ME/CFS" migraine \
    -k re -H PM -se RE_stdErr -es RE_Effect --outfile clr_metaaans_nest/SPC_cMD3_paper_two_layers.tsv \
    -qs RE_Effect_Qvalue RE_Effect_Qvalue RE_Effect_Qvalue RE_Effect_Qvalue \
    Qvalue Qvalue Qvalue Qvalue Qvalue Qvalue Qvalue Qvalue \
    -ps RE_Pvalue RE_Pvalue RE_Pvalue RE_Pvalue \
    RE_Pvalue RE_Pvalue RE_Pvalue RE_Pvalue RE_Pvalue RE_Pvalue RE_Pvalue RE_Pvalue \
    --sheets \
    clr_metaaans_nest/CRC_species_clr_metaanalysis.tsv clr_metaaans_nest/T2D_species_clr_metaanalysis.tsv clr_metaaans_nest/CD_species_clr_metaanalysis.tsv \
    clr_metaaans_nest/UC_species_clr_metaanalysis.tsv clr_metaaans_nest/ACVD_species_clr_metaanalysis.tsv clr_metaaans_nest/cirrhosis_species_clr_metaanalysis.tsv \
    clr_metaaans_nest/schizofrenia_species_clr_metaanalysis.tsv clr_metaaans_nest/asthma_species_clr_metaanalysis.tsv clr_metaaans_nest/STH_species_clr_metaanalysis.tsv \
    clr_metaaans_nest/BD_species_clr_metaanalysis.tsv clr_metaaans_nest/MECFS_species_clr_metaanalysis.tsv clr_metaaans_nest/migraine_species_clr_metaanalysis.tsv

python ../python_tools/hierarchical_metaanalysis.py \
     --names CRC T2D CD UC ACVD cirrhosis schizofrenia asthma STH BD "ME/CFS" migraine \
    -k re -H PM -se RE_stdErr -es RE_Effect --outfile clr_metaaans_nest/PWY_cMD3_paper_two_layers.tsv \
    -qs RE_Effect_Qvalue RE_Effect_Qvalue RE_Effect_Qvalue RE_Effect_Qvalue \
    Qvalue Qvalue Qvalue Qvalue Qvalue Qvalue Qvalue Qvalue \
    -ps RE_Pvalue RE_Pvalue RE_Pvalue RE_Pvalue \
    RE_Pvalue RE_Pvalue RE_Pvalue RE_Pvalue RE_Pvalue RE_Pvalue RE_Pvalue RE_Pvalue \
    --sheets \
    clr_metaaans_nest/CRC_pathways_clr_metaanalysis.tsv clr_metaaans_nest/T2D_pathways_clr_metaanalysis.tsv clr_metaaans_nest/CD_pathways_clr_metaanalysis.tsv \
    clr_metaaans_nest/UC_pathways_clr_metaanalysis.tsv clr_metaaans_nest/ACVD_pathways_clr_metaanalysis.tsv clr_metaaans_nest/cirrhosis_pathways_clr_metaanalysis.tsv \
    clr_metaaans_nest/schizofrenia_pathways_clr_metaanalysis.tsv clr_metaaans_nest/asthma_pathways_clr_metaanalysis.tsv clr_metaaans_nest/STH_pathways_clr_metaanalysis.tsv \
    clr_metaaans_nest/BD_pathways_clr_metaanalysis.tsv clr_metaaans_nest/MECFS_pathways_clr_metaanalysis.tsv clr_metaaans_nest/migraine_pathways_clr_metaanalysis.tsv

python ../python_tools/draw_figure_with_ma.py clr_metaaans_nest/SPC_cMD3_paper_two_layers.tsv --outfile images/SPC_FIG4_meta_10precent_prevalence -a relative_abundances/usable_all_species.tsv -re RE_Effect -x "Standardized mean difference" -es "_Effect" -qs "_Qvalue" --imp 30 --neg_max_rho 1.25 --pos_max_rho 1.25 --title "Meta-analysis of disease-associated microbial species" -ps "Unhealthy microbiome" -ns "Healthy microbiome" --legloc "lower right" -ar 0.05 -as 0.2 --color_red goldenrod --color_blue cornflowerblue --color_black black -rq RE_Qvalue --confint RE_conf_int --markers -ms 2 --boxes --legloc "upper left" 
 
python ../python_tools/draw_figure_with_ma.py clr_metaaans_nest/PWY_cMD3_paper_two_layers.tsv --outfile images/PWY_FIG4_meta_supp1 -a relative_abundances/usable_all_pathways.tsv -re RE_Effect -x "Standardized mean difference" -es "_Effect" -qs "_Qvalue" --imp 30 --neg_max_rho 1.0 --pos_max_rho 1.0 --title "Meta-analysis of disease-associated metabolic pathways" -ps "Unhealthy microbiome" -ns "Healthy microbiome" --legloc "lower right" -ar 0.05 -as 0.2 --color_red goldenrod --color_blue cornflowerblue --color_black black -rq RE_Qvalue --confint RE_conf_int --markers -ms 2 --boxes --legloc "upper left" #-mp 0.01 -mna 5

## SECTION 3: MACHINE LEARNING ON DISEASES
cd ML
python ml_tests_on_diseases_rf.py
python ../figure4_complete_ml.py 

## SECTION 4: ORAL INTROGRESSION ANALYSIS AND META-ANALYSIS
cd oral_introgression

python oral_introgression.py Oral_Richness ## DISEASES
python oral_introgression.py Oral_Entropy
python oral_introgression.py Oral_Richness -la age -tm AGE -dt tables/usable_for_age_species.tsv
python oral_introgression.py Oral_Richness -la BMI -tm REG -dt tables/usable_for_BMI_species.tsv
python oral_introgression.py Oral_Richness -la gender:male:female -tm SEX -dt tables/usable_for_sex_species.tsv
