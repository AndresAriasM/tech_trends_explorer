import Link from "next/link";
import { TrendingUp, History, FolderOpen, ArrowRight } from "lucide-react";

export default function HomePage() {
  return (
    <div className="p-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-10">
        <h1 className="text-3xl font-bold text-white mb-2">
          Tech Trends Explorer
        </h1>
        <p className="text-gray-400">
          Análisis de tendencias tecnológicas con el marco del Hype Cycle de Gartner.
          Universidad Pontificia Bolivariana · UPB 2036
        </p>
      </div>

      {/* Feature cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-10">
        <FeatureCard
          href="/hype-cycle"
          icon={TrendingUp}
          title="Hype Cycle"
          description="Analiza la posición de una tecnología en el ciclo de Gartner usando fuentes web y academicas."
          badge="Disponible"
          badgeColor="green"
        />
        <FeatureCard
          href="/history"
          icon={History}
          title="Historial"
          description="Consulta y reutiliza análisis previos almacenados en DynamoDB para evitar consumo de API."
          badge="Disponible"
          badgeColor="green"
        />
        <FeatureCard
          href="/categories"
          icon={FolderOpen}
          title="Categorías"
          description="Organiza las tecnologías analizadas en categorías jerárquicas para facilitar su seguimiento."
          badge="Disponible"
          badgeColor="green"
        />
      </div>

      {/* Coming soon */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">
          Próximamente
        </p>
        <div className="flex gap-6 text-sm text-gray-500">
          <span>📉 Curvas S (Scopus)</span>
          <span>🔍 Búsqueda de Tendencias</span>
          <span>📊 Dashboard unificado</span>
        </div>
      </div>
    </div>
  );
}

interface FeatureCardProps {
  href: string;
  icon: React.ElementType;
  title: string;
  description: string;
  badge: string;
  badgeColor: "green" | "yellow" | "gray";
}

function FeatureCard({
  href,
  icon: Icon,
  title,
  description,
  badge,
  badgeColor,
}: FeatureCardProps) {
  const badgeStyles = {
    green: "bg-green-500/10 text-green-400 border-green-500/20",
    yellow: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
    gray: "bg-gray-700 text-gray-400 border-gray-600",
  };

  return (
    <Link
      href={href}
      className="group block bg-gray-900 border border-gray-800 rounded-xl p-5 hover:border-blue-500/50 transition-colors"
    >
      <div className="flex items-start justify-between mb-3">
        <div className="p-2 bg-blue-500/10 rounded-lg">
          <Icon className="w-5 h-5 text-blue-400" />
        </div>
        <span
          className={`text-xs border rounded-full px-2 py-0.5 ${badgeStyles[badgeColor]}`}
        >
          {badge}
        </span>
      </div>
      <h3 className="font-semibold text-white mb-1">{title}</h3>
      <p className="text-xs text-gray-400 leading-relaxed mb-3">{description}</p>
      <div className="flex items-center gap-1 text-xs text-blue-400 group-hover:gap-2 transition-all">
        Abrir <ArrowRight className="w-3 h-3" />
      </div>
    </Link>
  );
}
