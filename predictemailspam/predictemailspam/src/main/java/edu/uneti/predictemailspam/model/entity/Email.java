package edu.uneti.predictemailspam.model.entity;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import jakarta.persistence.*;

import java.io.Serializable;

@Entity
@JsonIgnoreProperties({"hibernateLazyInitializer", "handler"})
public class Email implements Serializable {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;
    private String subject;
    @Column(columnDefinition = "TEXT")
    @Lob
    private String description;
    private String time;
    private Boolean isSpam;

    public Email() {
    }

    public Email(Integer id, String subject, String description, String time, Boolean isSpam) {
        this.id = id;
        this.subject = subject;
        this.description = description;
        this.time = time;
        this.isSpam = isSpam;
    }

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getSubject() {
        return subject;
    }

    public void setSubject(String title) {
        this.subject = title;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getTime() {
        return time;
    }

    public void setTime(String time) {
        this.time = time;
    }

    public Boolean getIsSpam() {
        return isSpam;
    }

    public void setIsSpam(Boolean isSpam) {
        this.isSpam = isSpam;
    }

    @Override
    public String toString() {
        return "Email{" +
                "id=" + id +
                ", subject='" + subject + '\'' +
                ", description='" + description + '\'' +
                ", time='" + time + '\'' +
                ", isSpam='" + isSpam + '\'' +
                '}';
    }
}
