package edu.uneti.predictemailspam.model.mapper;


import edu.uneti.predictemailspam.model.dto.EmailDto;
import edu.uneti.predictemailspam.model.entity.Email;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.util.ArrayList;
import java.util.List;

@SpringBootApplication
public class EmailMapper {

    public EmailDto toDto(Email email) {
        EmailDto emailDto = new EmailDto();
        emailDto.setId(email.getId());
        emailDto.setSubject(email.getSubject());
        emailDto.setDescription(email.getDescription());
        emailDto.setTime(email.getTime());
        emailDto.setSpam(email.getIsSpam());
        return emailDto;
    }

    public Email toEmail(EmailDto emailDto) {
        Email email = new Email();
        email.setId(emailDto.getId());
        email.setSubject(emailDto.getSubject());
        email.setDescription(emailDto.getDescription());
        email.setTime(emailDto.getTime());
        email.setIsSpam(emailDto.isSpam());
        return email;
    }

    public List<EmailDto> toEmailList(List<Email> emailList) {
        List<EmailDto> emailDtoList = new ArrayList<>();
        for (Email email : emailList) {
            emailDtoList.add(toDto(email));
        }
        return emailDtoList;
    }
}
