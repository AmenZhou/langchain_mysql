-- Create test tables
CREATE TABLE active_messages (
    id INT PRIMARY KEY AUTO_INCREMENT,
    message_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE message_participants (
    id INT PRIMARY KEY AUTO_INCREMENT,
    message_id INT,
    participant_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE message_room_trigger_relations (
    id INT PRIMARY KEY AUTO_INCREMENT,
    message_room_id INT,
    trigger_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE message_rooms (
    id INT PRIMARY KEY AUTO_INCREMENT,
    room_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE system_messages (
    id INT PRIMARY KEY AUTO_INCREMENT,
    message_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
); 
