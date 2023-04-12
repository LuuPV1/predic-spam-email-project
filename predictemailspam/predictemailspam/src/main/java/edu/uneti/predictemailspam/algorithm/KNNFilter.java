package edu.uneti.predictemailspam.algorithm;

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
        try (BufferedReader reader = Files.newBufferedReader(path)) {
            String line;

            while ((line = reader.readLine()) != null) {
                String[] data = getDataEmail(line);
                emails.add(data[0]);
                labels.add(data[1]);
                String[] words = data[0].split(" ");
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
        String[] words = content.split(" ");

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
        return "S".equals(maxLabel);
    }

    private static String[] getDataEmail(String content){
        int length = content.length();
        String[] data = new String[2];
        data[0] = content.substring(0, length - 1);
        data[1] = content.substring(length -1 );
        return data;
    }
}
