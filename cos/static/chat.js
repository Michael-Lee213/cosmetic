document.addEventListener('DOMContentLoaded', function() {
    const chatbotToggle = document.getElementById("chatbot-toggle");
    const chatbot = document.getElementById("chat-container");
    const chatClose = document.getElementById("chat-close");
    const sendBtn = document.getElementById("send-btn");
    const chatInput = document.getElementById("chat-input");
    const chatBody = document.getElementById("chat-body");
    

    // 챗봇 열기/닫기 기능
    chatbotToggle.addEventListener("click", function() {
        chatbot.style.display = chatbot.style.display === "block" ? "none" : "block";
    });

    chatClose.addEventListener("click", function() {
        chatbot.style.display = "none";
    });

    // 메시지 전송 기능
    sendBtn.addEventListener("click", sendMessage);
    chatInput.addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
            sendMessage();
        }
    });

    function sendMessage() {
        let message = chatInput.value.trim();
        if (message === "") return;

        // 사용자 메시지 추가 (🙋‍♂️ 이모티콘 포함)
        chatBody.innerHTML += `<div class='user-message'>🙋‍♂️ ${message}</div>`;
        chatBody.scrollTop = chatBody.scrollHeight;

        const startTime = performance.now(); // ✅ 반응 속도 측정 시작

        fetch("/ask", {
            method: "POST",
            body: JSON.stringify({ question: message }),
            headers: { "Content-Type": "application/json" }
        })
        .then(response => response.json())
        .then(data => {
            const endTime = performance.now(); // ✅ 반응 속도 측정 끝
            const responseTime = ((endTime - startTime) / 1000).toFixed(2);

            let botResponse = formatResponse(data.answer);
            chatBody.innerHTML += `<div class='bot-message'>🤖 ${botResponse}<br><small>⏳ 응답 시간: ${responseTime}s</small></div>`;
            chatBody.scrollTop = chatBody.scrollHeight;
        })
        .catch(error => {
            chatBody.innerHTML += `<div class='bot-message error'>🤖 오류 발생: ${error}</div>`;
        });

        chatInput.value = "";
    }

// ✅ 개조식 응답 적용 (가독성 개선)
function formatResponse(text) {
    return text
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>") // **텍스트** → 굵게 변환
        .replace(/- /g, "<br>🔹 ") // 개조식으로 변환
        .replace(/\n/g, "<br>") // 줄바꿈 적용
        .replace(/(\b[A-Z][a-z]+\b)/g, "<span style='color: #007bff; font-weight: bold;'>$1</span>"); // 키워드 강조
}
    

    
});
