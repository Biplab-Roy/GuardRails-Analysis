python "../guards/driver.py" -g beaver -d ../GPTeacher/Codegen.csv | tee beaver_codegen.out
python "../guards/driver.py" -g flan_t5_base -d ../GPTeacher/Codegen.csv | tee flan_t5_base_codegen.out
python "../guards/driver.py" -g text_moderator -d ../GPTeacher/Codegen.csv | tee text_moderator_codegen.out
python "../guards/driver.py" -g toxicity -d ../GPTeacher/Codegen.csv | tee toxicity_codegen.out
python "../guards/driver.py" -g distilroberta -d ../GPTeacher/Codegen.csv | tee distilroberta_codegen.out
python "../guards/driver.py" -g offensiveClassifier -d ../GPTeacher/Codegen.csv | tee offensiveClassifier_codegen.out
python "../guards/driver.py" -g hateClassifier -d ../GPTeacher/Codegen.csv | tee hateClassifier_codegen.out
python "../guards/driver.py" -g beaver -d ../GPTeacher/Instruct.csv | tee beaver_instruct.out
python "../guards/driver.py" -g flan_t5_base -d ../GPTeacher/Instruct.csv | tee flan_t5_base_instruct.out
python "../guards/driver.py" -g text_moderator -d ../GPTeacher/Instruct.csv | tee text_moderator_instruct.out
python "../guards/driver.py" -g toxicity -d ../GPTeacher/Instruct.csv | tee toxicity_instruct.out
python "../guards/driver.py" -g distilroberta -d ../GPTeacher/Instruct.csv | tee distilroberta_instruct.out
python "../guards/driver.py" -g offensiveClassifier -d ../GPTeacher/Instruct.csv | tee offensiveClassifier_instruct.out
python "../guards/driver.py" -g hateClassifier -d ../GPTeacher/Instruct.csv | tee hateClassifier_instruct.out
python "../guards/driver.py" -g beaver -d ../GPTeacher/Roleplay\ Supplemental.csv | tee beaver_roleplay_supplemental.out
python "../guards/driver.py" -g flan_t5_base -d ../GPTeacher/Roleplay\ Supplemental.csv | tee flan_t5_base_roleplay_supplemental.out
python "../guards/driver.py" -g text_moderator -d ../GPTeacher/Roleplay\ Supplemental.csv | tee text_moderator_roleplay_supplemental.out
python "../guards/driver.py" -g toxicity -d ../GPTeacher/Roleplay\ Supplemental.csv | tee toxicity_roleplay_supplemental.out
python "../guards/driver.py" -g distilroberta -d ../GPTeacher/Roleplay\ Supplemental.csv | tee distilroberta_roleplay_supplemental.out
python "../guards/driver.py" -g offensiveClassifier -d ../GPTeacher/Roleplay\ Supplemental.csv | tee offensiveClassifier_roleplay_supplemental.out
python "../guards/driver.py" -g hateClassifier -d ../GPTeacher/Roleplay\ Supplemental.csv | tee hateClassifier_roleplay_supplemental.out
python "../guards/driver.py" -g beaver -d ../GPTeacher/Toolformer.csv | tee beaver_toolformer.out
python "../guards/driver.py" -g flan_t5_base -d ../GPTeacher/Toolformer.csv | tee flan_t5_base_toolformer.out
python "../guards/driver.py" -g text_moderator -d ../GPTeacher/Toolformer.csv | tee text_moderator_toolformer.out
python "../guards/driver.py" -g toxicity -d ../GPTeacher/Toolformer.csv | tee toxicity_toolformer.out
python "../guards/driver.py" -g distilroberta -d ../GPTeacher/Toolformer.csv | tee distilroberta_toolformer.out
python "../guards/driver.py" -g offensiveClassifier -d ../GPTeacher/Toolformer.csv | tee offensiveClassifier_toolformer.out
python "../guards/driver.py" -g hateClassifier -d ../GPTeacher/Toolformer.csv | tee hateClassifier_toolformer.out
mv *.out ./outs