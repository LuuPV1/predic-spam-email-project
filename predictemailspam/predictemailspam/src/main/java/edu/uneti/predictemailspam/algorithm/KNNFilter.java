package edu.uneti.predictemailspam.service.impl;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class KNNFilter {
    private List<String> emails;
    private List<String> labels;
    private Map<String, Double> idfMap;

    public KNNFilter(String filePath) {
        emails = new ArrayList<>();
        labels = new ArrayList<>();
        idfMap = new HashMap<>();
        loadData(filePath);
    }

    private void loadData(String filePath) {
        Path path = Paths.get(filePath);
        try (BufferedReader reader = Files.newBufferedReader(path)){
            String line;

            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                emails.add(parts[0]);
                labels.add(parts[parts.length - 1]);
                String[] words = parts[0].split(" ");
                for (String word : words) {
                    if (!idfMap.containsKey(word)) {
                        idfMap.put(word, 1.0);
                    }
                }
            }
            for (String word : idfMap.keySet()) {
                double idf = 0;
                for (String email : emails) {
                    if (email.contains(word)) {
                        idf++;
                    }
                }
                idfMap.put(word, Math.log(emails.size() / (idf + 1)));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public boolean predict(String content) {
        String email = removeCommaContent(content);
        String[] words = removeRedundantCharacters(email);

        Map<String, Double> tfMap = new HashMap<>();
        for (String word : words) {
            if (!tfMap.containsKey(word)) {
                tfMap.put(word, 1.0);
            } else {
                tfMap.put(word, tfMap.get(word) + 1);
            }
        }
        double maxScore = 0;
        String maxLabel = "";
        for (int i = 0; i < emails.size(); i++) {
            String trainEmail = emails.get(i);
            String trainLabel = labels.get(i);
            String[] trainWords = trainEmail.split(" ");
            Map<String, Double> trainTfMap = new HashMap<>();
            for (String word : trainWords) {
                if (!trainTfMap.containsKey(word)) {
                    trainTfMap.put(word, 1.0);
                } else {
                    trainTfMap.put(word, trainTfMap.get(word) + 1);
                }
            }
            double score = 0;
            for (String word : words) {
                if (trainTfMap.containsKey(word)) {
                    score += tfMap.get(word) * trainTfMap.get(word) * idfMap.get(word);
                }
            }
            if (score > maxScore) {
                maxScore = score;
                maxLabel = trainLabel;
            }
        }
        return "spam".equals(maxLabel);
    }

    private String[] removeRedundantCharacters(String content) {
        String[] words = content.replaceAll("[^a-zA-Z0-9]", " ").split(" ");
        String[] result = new String[words.length];
        for (int i = 0; i < words.length; i++) {
            if(!words[i].isEmpty()){
                result[i] = words[i];
            }
        }
        return result;
    }

    private static String removeCommaContent(String content){
        return content.replaceAll(",", "");
    }
}
