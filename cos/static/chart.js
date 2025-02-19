async function fetchResponseTimeData() {
    try {
        // ✅ 1. 현재 모델의 응답 시간 가져오기
        const response = await fetch("/response_time_chart_json");
        const data = await response.json();

        if (data.error) {
            console.error("🚨 현재 모델 응답 데이터 오류:", data.error);
            return;
        }

        // ✅ 2. 이전 모델의 응답 시간 가져오기
        const prevResponse = await fetch("/previous_response_time_json");
        const prevData = await prevResponse.json();

        // ✅ 콘솔에서 API 응답 데이터 확인
        console.log("✅ 현재 모델 데이터 응답:", data);
        console.log("✅ 이전 모델 데이터 응답:", prevData);

        // 🚨 이전 모델 데이터가 올바르게 반환되지 않았을 경우
        let previousResponseTimes = [];
        if (prevData && prevData.previous_response_times) {
            // 이전 모델 데이터가 존재하는 경우
            previousResponseTimes = prevData.previous_response_times;
        } else {
            console.error("🚨 이전 모델 응답 데이터 오류:", prevData.error || "알 수 없는 오류");
        }

        // ✅ 3. X축 라벨 생성 (횟수)
        const labels = data.response_times.map((_, index) => index + 1);

        // ✅ 4. 데이터 개수 불일치 처리 (이전 모델 데이터가 적을 경우 보정)
        if (previousResponseTimes.length < data.response_times.length) {
            const lastValue = previousResponseTimes.length > 0 ? previousResponseTimes[previousResponseTimes.length - 1] : 0;
            previousResponseTimes = previousResponseTimes.concat(
                new Array(data.response_times.length - previousResponseTimes.length).fill(lastValue)
            );
        }

        // ✅ 5. Chart.js 설정 및 그래프 생성
        const ctx = document.getElementById("responseTimeChart").getContext("2d");

        new Chart(ctx, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "현재 모델 응답 시간 (초)",
                        data: data.response_times,
                        borderColor: "#4CAF50", // 기존 색상
                        backgroundColor: "rgba(76, 175, 80, 0.4)", // 라인 아래에 그라데이션 추가
                        borderWidth: 3,
                        tension: 0.4, // 라인을 부드럽게 만들어 줍니다
                        pointRadius: 5, // 포인트 크기 설정
                        pointBackgroundColor: "#4CAF50", // 포인트 색상
                        pointBorderWidth: 2,
                        pointHoverRadius: 7, // 포인트 hover 시 크기 변경
                        pointHoverBackgroundColor: "#388E3C", // hover 시 포인트 색상
                        fill: true, // 라인 아래 영역을 채우는 옵션
                    },
                    {
                        label: "이전 모델 응답 시간 (초)",
                        data: previousResponseTimes,
                        borderColor: "#FF9800", // 기존 색상
                        backgroundColor: "rgba(255, 152, 0, 0.4)", // 라인 아래에 그라데이션 추가
                        borderWidth: 3,
                        tension: 0.4, // 라인을 부드럽게 만들어 줍니다
                        pointRadius: 5, // 포인트 크기 설정
                        pointBackgroundColor: "#FF9800", // 포인트 색상
                        pointBorderWidth: 2,
                        pointHoverRadius: 7, // 포인트 hover 시 크기 변경
                        pointHoverBackgroundColor: "#F57C00", // hover 시 포인트 색상
                        fill: true, // 라인 아래 영역을 채우는 옵션
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    
                    legend: {                    
                        position: "top", // 왼쪽 상단에 위치하게 설정
                        align: "start", // 왼쪽 정렬
                        labels: {
                            font: {
                                size: 14,
                                weight: "bold"
                            },
                            color: "#333"
                        }
                    },
                    tooltip: {
                        backgroundColor: "rgba(0, 0, 0, 0.7)", // 툴팁 배경 색상
                        titleFont: {
                            size: 14,
                            weight: "bold"
                        },
                        bodyFont: {
                            size: 12
                        },
                        footerFont: {
                            size: 12
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: "횟수 (회)",
                            color: "#333",
                            font: {
                                size: 16,
                                weight: "bold"
                            }
                        },
                        ticks: {
                            autoSkip: true,   // 자동으로 간격 조절
                            maxRotation: 0,
                            font: {
                                size: 12
                            },
                            color: "#333"
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: "응답 시간 (초)",
                            color: "#333",
                            font: {
                                size: 16,
                                weight: "bold"
                            }
                        },
                        ticks: {
                            font: {
                                size: 12
                            },
                            color: "#333"
                        }
                    }
                }
            }
        });

    } catch (error) {
        console.error("🚨 데이터 가져오기 실패:", error);
    }
}

// 🔹 실행
fetchResponseTimeData();
