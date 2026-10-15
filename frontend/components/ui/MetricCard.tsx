"use client";

interface Props {
  label: string;
  value: string | number;
  sub?: string;
  color?: "blue" | "green" | "yellow" | "red" | "purple" | "default";
}

const colorMap = {
  blue: "border-blue-500/40 bg-blue-500/5 text-blue-400",
  green: "border-green-500/40 bg-green-500/5 text-green-400",
  yellow: "border-yellow-500/40 bg-yellow-500/5 text-yellow-400",
  red: "border-red-500/40 bg-red-500/5 text-red-400",
  purple: "border-purple-500/40 bg-purple-500/5 text-purple-400",
  default: "border-gray-700 bg-gray-800 text-gray-300",
};

export function MetricCard({ label, value, sub, color = "default" }: Props) {
  return (
    <div className={`border rounded-xl p-4 ${colorMap[color]}`}>
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className={`text-2xl font-bold ${colorMap[color].split(" ")[2]}`}>
        {value}
      </p>
      {sub && <p className="text-xs text-gray-500 mt-0.5">{sub}</p>}
    </div>
  );
}
