/* src/App.jsx */
import { useState } from "react";
import axios from "axios";

/* описание полей: [имя, типInput] */
const fields = [
  ["income", "number"],
  ["name_email_similarity", "number"],
  ["prev_address_months_count", "number"],
  ["current_address_months_count", "number"],
  ["customer_age", "number"],
  ["days_since_request", "number"],
  ["intended_balcon_amount", "number"],
  ["payment_type", "text"],
  ["zip_count_4w", "number"],
  ["velocity_6h", "number"],
  ["velocity_24h", "number"],
  ["velocity_4w", "number"],
  ["bank_branch_count_8w", "number"],
  ["date_of_birth_distinct_emails_4w", "number"],
  ["employment_status", "text"],
  ["credit_risk_score", "number"],
  ["email_is_free", "number"],
  ["housing_status", "text"],
  ["phone_home_valid", "number"],
  ["phone_mobile_valid", "number"],
  ["bank_months_count", "number"],
  ["has_other_cards", "number"],
  ["proposed_credit_limit", "number"],
  ["foreign_request", "number"],
  ["source", "text"],
  ["session_length_in_minutes", "number"],
  ["device_os", "text"],
  ["keep_alive_session", "number"],
  ["device_distinct_emails_8w", "number"],
  ["device_fraud_count", "number"],
  ["month", "number"],
];

/* ▶︎ значения по умолчанию  ──  копия твоего предыдущего payload */
const defaultValues = {
  income: 0.3,
  name_email_similarity: 0.98,
  prev_address_months_count: -1,
  current_address_months_count: 25,
  customer_age: 40,
  days_since_request: 0.006,
  intended_balcon_amount: 102.45,
  payment_type: "AA",
  zip_count_4w: 1059,
  velocity_6h: 13096.03,
  velocity_24h: 7850.95,
  velocity_4w: 6742.08,
  bank_branch_count_8w: 5,
  date_of_birth_distinct_emails_4w: 5,
  employment_status: "CB",
  credit_risk_score: 163,
  email_is_free: 1,
  housing_status: "BC",
  phone_home_valid: 0,
  phone_mobile_valid: 1,
  bank_months_count: 9,
  has_other_cards: 0,
  proposed_credit_limit: 1500.0,
  foreign_request: 0,
  source: "INTERNET",
  session_length_in_minutes: 16.22,
  device_os: "linux",
  keep_alive_session: 1,
  device_distinct_emails_8w: 1,
  device_fraud_count: 0,
  month: 0,
};

export default function App() {
  const [data, setData] = useState(defaultValues);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  /* единый onChange */
  const handleChange = (e) =>
    setData((d) => ({ ...d, [e.target.name]: e.target.value }));

  /* submit */
  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const payload = {};
    fields.forEach(([k, t]) => {
      payload[k] = t === "number" ? Number(data[k]) : data[k];
    });

    try {
      const { data: res } = await axios.post("http://localhost:8000/predict", payload);
      setResult(res); // {prediction, probability, threshold_used}
    } catch (err) {
      alert(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  /* «страница» с результатом */
  if (result) {
    return (
      <main className="wrapper">
        <h1>Результат</h1>
        <p><b>Prediction:</b> {result.prediction}</p>
        <p><b>Probability:</b> {result.probability.toFixed(4)}</p>
        <p><b>Threshold used:</b> {result.threshold_used}</p>
        <button onClick={() => setResult(null)}>← Назад</button>
      </main>
    );
  }

  /* форма в сетке 4×N */
  return (
    <main className="wrapper">
      <h1>Fraud Prediction</h1>

      <form onSubmit={submit} className="grid-form">
        {fields.map(([name, type]) => (
          <label key={name} className="field">
            <span>{name.replace(/_/g, " ")}</span>
            <input
              type={type}
              step={type === "number" ? "any" : undefined}
              name={name}
              value={data[name]}
              onChange={handleChange}
              required
            />
          </label>
        ))}

        <div className="buttons">
          <button type="submit" disabled={loading}>
            {loading ? "Predicting…" : "Predict"}
          </button>
        </div>
      </form>
    </main>
  );
}
